"""Camera capture and device arguments shared by the Cubie demos."""

import argparse
import sys
import time
from threading import Condition, Event, Thread

import cv2
from calibration_common import IMAGE_SIZE

from smartcane.sensors.bno085 import Bno085Stream

DEVICE = 1 if sys.platform == "win32" else 0
FPS = 25
PIXEL_FORMAT = "MJPG"


def open_camera(camera):
    if isinstance(camera, str) and camera.isdecimal():
        camera = int(camera)
    backend = {"win32": cv2.CAP_DSHOW, "linux": cv2.CAP_V4L2}.get(
        sys.platform, cv2.CAP_ANY
    )
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera: {camera}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PIXEL_FORMAT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, FPS)
        if backend == cv2.CAP_V4L2:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {camera}: {width:g}x{height:g}, FPS={fps:g}", flush=True)
        return cap
    except BaseException:
        cap.release()
        raise


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read camera frame")
    if frame.ndim == 3 and frame.shape[2] == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    if frame.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(f"Unexpected camera frame: {frame.shape}")
    return frame


def gpio_location(value):
    chip, sep, offset = value.rpartition(":")
    if not sep or not chip.startswith("/dev/gpiochip") or not offset.isdecimal():
        raise argparse.ArgumentTypeError("Use /dev/gpiochipN:OFFSET for GPIO lines")
    return chip, int(offset)


class CameraFrames:
    """Drain camera frames continuously so low-rate preview does not age the frames."""

    def __init__(self, camera):
        self.camera = camera
        self.changed = Condition()
        self.stopped = Event()
        self.frame = None
        self.failure = None
        self.thread = Thread(target=self.read, daemon=True)

    def __enter__(self):
        self.cap = open_camera(self.camera)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join()
        self.cap.release()

    def read(self):
        try:
            while not self.stopped.is_set():
                raw = read_frame(self.cap)
                stamp = time.monotonic_ns()
                with self.changed:
                    self.frame = (stamp, raw)
                    self.changed.notify_all()
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            with self.changed:
                self.failure = exc
                self.changed.notify_all()

    def latest(self, after_ns=-1):
        with self.changed:
            self.changed.wait_for(
                lambda: (
                    self.failure
                    or (self.frame is not None and self.frame[0] > after_ns)
                ),
                timeout=3,
            )
            if self.failure:
                raise RuntimeError(f"Camera: {self.failure}") from self.failure
            if (
                self.frame is None
                or self.frame[0] <= after_ns
                or time.monotonic_ns() - self.frame[0] > 1_000_000_000
            ):
                raise RuntimeError("Camera supplied no recent frame")
            return self.frame


def add_imu_args(parser):
    parser.add_argument("--spi-device", default="/dev/spidev1.0")
    parser.add_argument("--int-line", type=gpio_location, default="/dev/gpiochip0:312")
    parser.add_argument("--rst-line", type=gpio_location, default="/dev/gpiochip0:313")
    parser.add_argument("--wake-line", type=gpio_location, default="/dev/gpiochip1:5")


def imu_stream(args):
    return Bno085Stream(args.spi_device, args.int_line, args.rst_line, args.wake_line)
