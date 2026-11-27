"""Read BNO085 game rotation vectors over Linux SPI.

Samples contain a quaternion in (w, x, y, z) order and a host monotonic
timestamp recorded after reading the SPI packet.
"""

import struct
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Condition, Event, Thread

import numpy as np

PERIOD_MS = 20
WINDOW_NS = 200_000_000
MAX_NEAREST_NS = 60_000_000
MAX_SPREAD_DEG = 1.0


def feature_packet():
    payload = struct.pack("<BBBBBIII", 0xFD, 0x08, 0, 0, 0, PERIOD_MS * 1000, 0, 0)
    return struct.pack("<HBB", 4 + len(payload), 2, 0) + payload


def decode_packet(wire):
    if len(wire) < 4:
        return None
    wire = bytes(wire)
    length_raw, channel, _ = struct.unpack_from("<HBB", wire)
    length = length_raw & 0x7FFF
    if not 4 <= length <= len(wire):
        return None
    return channel, bool(length_raw & 0x8000), wire[4:length]


def rotation_report(packet):
    if packet is None:
        return None
    channel, continuation, payload = packet
    if channel != 3 or continuation or len(payload) < 17:
        return None
    if payload[0] != 0xFB or payload[5] != 0x08:
        return None
    x, y, z, w = struct.unpack_from("<hhhh", payload, 9)
    quaternion = np.array([w, x, y, z], dtype=float) / (1 << 14)
    norm = np.linalg.norm(quaternion)
    if not 0.75 <= norm <= 1.25:
        return None
    return quaternion / norm, payload[7] & 0x03


def request_gpio(gpiod, location, direction, **kwargs):
    chip, offset = location
    settings = gpiod.LineSettings(direction=direction, **kwargs)
    return gpiod.request_lines(
        chip, consumer="smartcane-bno085", config={offset: settings}
    )


def wait_int_low(request, location, timeout, inactive):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if request.get_value(location[1]) == inactive:
            return True
        time.sleep(0.001)
    return False


@dataclass(frozen=True)
class ImuSample:
    host_ns: int
    quaternion: np.ndarray
    status: int


@dataclass(frozen=True)
class ImuMatch:
    quaternion: np.ndarray
    spread_deg: float
    samples: list[ImuSample]
    nearest_ns: int


def match_samples(samples, frame_ns):
    nearby = [
        sample for sample in samples if abs(sample.host_ns - frame_ns) <= WINDOW_NS
    ]
    if len(nearby) < 8 or not nearby[0].host_ns < frame_ns < nearby[-1].host_ns:
        raise ValueError("Waiting for IMU samples around the camera frame")
    nearest_ns = min(abs(sample.host_ns - frame_ns) for sample in nearby)
    if nearest_ns > MAX_NEAREST_NS:
        raise ValueError("IMU samples are too far from the camera frame time")
    quaternions = np.stack([sample.quaternion for sample in nearby])
    quaternions[quaternions @ quaternions[0] < 0] *= -1
    mean = quaternions.mean(axis=0)
    mean /= np.linalg.norm(mean)
    farthest = np.min(np.abs(quaternions @ mean))
    spread = float(np.degrees(2 * np.arccos(np.clip(farthest, 0.0, 1.0))))
    if spread > MAX_SPREAD_DEG:
        raise ValueError(f"Hold the cane still. IMU spread is {spread:.2f} deg.")
    return ImuMatch(mean, spread, nearby, nearest_ns)


class Bno085Stream:
    """Own a BNO085 SPI connection and buffer timestamped orientation samples."""

    def __init__(self, device, int_line, rst_line, wake_line):
        if len({int_line, rst_line, wake_line}) != 3:
            raise ValueError("INT, RST and WAKE need distinct GPIO lines")
        self.device = device
        self.int_line = int_line
        self.rst_line = rst_line
        self.wake_line = wake_line
        self.samples = deque(maxlen=150)
        self.changed = Condition()
        self.stopped = Event()
        self.failure = None
        self.thread = Thread(target=self.read, daemon=True)

    def __enter__(self):
        self.thread.start()
        try:
            with self.changed:
                self.changed.wait_for(lambda: self.samples or self.failure, timeout=7)
                if self.failure:
                    raise RuntimeError(f"BNO085: {self.failure}") from self.failure
                if not self.samples:
                    raise RuntimeError(
                        "BNO085 supplied no rotation vector. Check wiring."
                    )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.stopped.set()
        self.thread.join()

    def match(self, frame_ns):
        with self.changed:
            self.changed.wait_for(
                lambda: (
                    self.failure or self.samples[-1].host_ns >= frame_ns + WINDOW_NS
                ),
                timeout=0.45,
            )
            if self.failure:
                raise RuntimeError(f"BNO085: {self.failure}") from self.failure
            return match_samples(self.samples, frame_ns)

    def read(self):
        try:
            self.read_device()
        except Exception as exc:
            with self.changed:
                self.failure = exc
                self.changed.notify_all()

    def read_device(self):
        try:
            import gpiod
            import spidev
            from gpiod.line import Bias, Direction, Value
        except ImportError as exc:
            raise RuntimeError(
                "Install the IMU extra with uv sync --locked --extra imu"
            ) from exc

        if not self.device.startswith("/dev/spidev"):
            raise ValueError("Use a SPI device such as /dev/spidev1.0")
        try:
            bus, chip_select = map(
                int, self.device.removeprefix("/dev/spidev").split(".")
            )
        except ValueError as exc:
            raise ValueError("Use a SPI device such as /dev/spidev1.0") from exc

        with ExitStack() as stack:
            try:
                interrupt = request_gpio(
                    gpiod, self.int_line, Direction.INPUT, bias=Bias.PULL_UP
                )
            except OSError:
                interrupt = request_gpio(gpiod, self.int_line, Direction.INPUT)
            stack.enter_context(interrupt)
            wake = stack.enter_context(
                request_gpio(
                    gpiod, self.wake_line, Direction.OUTPUT, output_value=Value.ACTIVE
                )
            )
            reset = stack.enter_context(
                request_gpio(
                    gpiod, self.rst_line, Direction.OUTPUT, output_value=Value.ACTIVE
                )
            )
            spi = spidev.SpiDev()
            spi.open(bus, chip_select)
            stack.callback(spi.close)
            spi.mode, spi.max_speed_hz, spi.bits_per_word = 3, 1_000_000, 8

            time.sleep(0.03)
            for value in (Value.ACTIVE, Value.INACTIVE, Value.ACTIVE):
                reset.set_value(self.rst_line[1], value)
                time.sleep(0.02)
            time.sleep(0.15)

            boot_packets = 0
            deadline = time.monotonic() + 3
            while (
                time.monotonic() < deadline
                and boot_packets < 12
                and not self.stopped.is_set()
            ):
                if wait_int_low(interrupt, self.int_line, 0.15, Value.INACTIVE):
                    if decode_packet(spi.xfer2([0] * 512)) is None:
                        break
                    boot_packets += 1
                    time.sleep(0.002)
            if not boot_packets:
                raise RuntimeError(
                    "No SHTP packets after reset. Check SPI and GPIO wiring."
                )

            try:
                if not wait_int_low(interrupt, self.int_line, 0.2, Value.INACTIVE):
                    wake.set_value(self.wake_line[1], Value.INACTIVE)
                    if not wait_int_low(interrupt, self.int_line, 1.0, Value.INACTIVE):
                        raise RuntimeError("INT did not go low after asserting WAKE")
                spi.xfer2(list(feature_packet()))
            finally:
                wake.set_value(self.wake_line[1], Value.ACTIVE)

            while not self.stopped.is_set():
                if not wait_int_low(interrupt, self.int_line, 0.1, Value.INACTIVE):
                    continue
                packet = decode_packet(spi.xfer2([0] * 512))
                stamp = time.monotonic_ns()
                report = rotation_report(packet)
                if report is not None:
                    with self.changed:
                        self.samples.append(ImuSample(stamp, *report))
                        self.changed.notify_all()
                time.sleep(0.001)
