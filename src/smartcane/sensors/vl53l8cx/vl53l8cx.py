"""Read 8x8 VL53L8CX ranges over Linux I2C with the native ST driver."""

from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class VL53L8CXError(RuntimeError):
    """ST ULD/C-shim operation failed; includes native step and status."""


class _CFrame(C.Structure):
    # Must match native/tof_shim.h, ABI 1, without packing changes.
    _fields_ = [
        ("distance_mm", C.c_int16 * 64),
        ("target_status", C.c_uint8 * 64),
        ("target_count", C.c_uint8 * 64),
        ("streamcount", C.c_uint8),
        ("resolution", C.c_uint8),
        ("reserved", C.c_uint8 * 6),
        ("timestamp_ns", C.c_uint64),
    ]


@dataclass(frozen=True)
class ToFFrame:
    """One 8x8 ranging result. Raw arrays are copied out of native memory."""

    distance_mm: np.ndarray  # int16, shape (8, 8); raw values, including invalid zones
    target_status: np.ndarray  # uint8, shape (8, 8)
    target_count: np.ndarray  # uint8, shape (8, 8)
    timestamp_ns: int  # host monotonic clock at read completion
    streamcount: int  # 8-bit ST stream count, wraps around

    @property
    def valid_mask(self) -> np.ndarray:
        """High-confidence returns: status == 5, count > 0, distance > 0."""
        return (
            (self.target_status == 5) & (self.target_count > 0) & (self.distance_mm > 0)
        )

    @property
    def tentative_mask(self) -> np.ndarray:
        """Lower-confidence returns with status 6 or 9 and a positive range."""
        return (
            np.isin(self.target_status, (6, 9))
            & (self.target_count > 0)
            & (self.distance_mm > 0)
        )

    def distances_for_mapping(self, *, include_tentative: bool = False) -> np.ndarray:
        """Return float32 ranges with NaN for zones without an accepted return."""
        mask = self.valid_mask
        if include_tentative:
            mask = mask | self.tentative_mask
        return np.where(mask, self.distance_mm, np.nan).astype(np.float32)


class VL53L8CX:
    """Own one Linux VL53L8CX I2C device via native ST ULD.

    Example::
        with VL53L8CX(bus=7) as tof:
            tof.start(resolution=64, frequency_hz=10)
            frame = tof.read_frame(timeout_ms=1200)
            if frame is not None:
                print(frame.distance_mm, frame.valid_mask)

    Use one reader per sensor. The driver polls I2C for data readiness.
    """

    def __init__(
        self, bus: int | str | Path = 7, *, lib_path: str | Path | None = None
    ):
        if isinstance(bus, int):
            if bus < 0:
                raise ValueError("I2C bus must be nonnegative")
            bus_path = f"/dev/i2c-{bus}"
        else:
            bus_path = str(bus)
            if not bus_path.startswith("/dev/i2c-"):
                raise ValueError("bus must be an I2C number or /dev/i2c-N path")
        self.bus = bus_path
        if lib_path is None:
            lib_path = (
                Path(__file__).resolve().parent / "native" / "build" / "libvl53l8cx.so"
            )
        lib_path = Path(lib_path).expanduser().resolve()
        if not lib_path.is_file():
            raise FileNotFoundError(
                f"Missing native library: {lib_path}. "
                "Build with make -C src/smartcane/sensors/vl53l8cx/native"
            )
        self._lib = C.CDLL(str(lib_path), use_errno=True)
        self._configure_signatures()
        version = self._lib.tof_abi_version()
        if version != 1:
            raise VL53L8CXError(f"C shim ABI mismatch: expected 1, got {version}")
        if C.sizeof(_CFrame) != 272 or _CFrame.timestamp_ns.offset != 264:
            raise VL53L8CXError("ctypes tof_frame layout mismatch with ABI v1")
        C.set_errno(0)
        self._device = self._lib.tof_open(os.fsencode(self.bus))
        if not self._device:
            err = C.get_errno()
            if err:
                raise OSError(err, os.strerror(err), self.bus)
            raise VL53L8CXError(f"tof_open({self.bus}) failed")
        self._initialized = False
        self._started = False

    def _configure_signatures(self) -> None:
        lib = self._lib
        lib.tof_abi_version.argtypes = []
        lib.tof_abi_version.restype = C.c_int
        lib.tof_open.argtypes = [C.c_char_p]
        lib.tof_open.restype = C.c_void_p
        lib.tof_initialize.argtypes = [C.c_void_p]
        lib.tof_initialize.restype = C.c_int
        lib.tof_start.argtypes = [C.c_void_p, C.c_uint]
        lib.tof_start.restype = C.c_int
        lib.tof_read.argtypes = [C.c_void_p, C.POINTER(_CFrame), C.c_uint]
        lib.tof_read.restype = C.c_int
        lib.tof_stop.argtypes = [C.c_void_p]
        lib.tof_stop.restype = C.c_int
        lib.tof_close.argtypes = [C.c_void_p]
        lib.tof_close.restype = None
        lib.tof_last_step.argtypes = [C.c_void_p]
        lib.tof_last_step.restype = C.c_char_p
        lib.tof_last_status.argtypes = [C.c_void_p]
        lib.tof_last_status.restype = C.c_int

    def _require_open(self) -> None:
        if not self._device:
            raise VL53L8CXError("VL53L8CX is closed")

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        raw = self._lib.tof_last_step(self._device)
        step = raw.decode("utf-8", "replace") if raw else ""
        status = self._lib.tof_last_status(self._device)
        raise VL53L8CXError(
            f"{operation} failed at {step or operation} (ST status={status})"
        )

    def initialize(self) -> None:
        """Initialize device and upload firmware. Can take several seconds."""
        self._require_open()
        if not self._initialized:
            self._check(self._lib.tof_initialize(self._device), "initialize")
            self._initialized = True

    def start(self, *, resolution: int = 64, frequency_hz: int = 10) -> None:
        """Start continuous ranging. ABI v1 supports 8x8 at 1..15 Hz."""
        self._require_open()
        if resolution != 64:
            raise ValueError("This C shim supports only resolution=64 (8x8)")
        if not isinstance(frequency_hz, int) or not 1 <= frequency_hz <= 15:
            raise ValueError("frequency_hz must be an integer from 1 through 15")
        if self._started:
            raise VL53L8CXError("Ranging is already started")
        self.initialize()
        self._check(self._lib.tof_start(self._device, frequency_hz), "start")
        self._started = True

    def read_frame(self, timeout_ms: int = 1200) -> ToFFrame | None:
        """Return a copied frame, or None if the read times out."""
        self._require_open()
        if not self._started:
            raise VL53L8CXError("Call start() before read_frame()")
        if not isinstance(timeout_ms, int) or not 0 <= timeout_ms <= 60000:
            raise ValueError("timeout_ms must be an integer in [0, 60000]")
        c_frame = _CFrame()
        rc = self._lib.tof_read(self._device, C.byref(c_frame), timeout_ms)
        if rc == 0:
            return None
        if rc != 1:
            self._check(rc, "read_frame")
            raise VL53L8CXError(f"Unexpected tof_read result {rc}")
        if c_frame.resolution != 64:
            raise VL53L8CXError(
                f"Unexpected resolution in C frame: {c_frame.resolution}"
            )

        # Copy before leaving: these arrays must not reference a native temporary.
        def to_array(field: str, dtype: np.dtype) -> np.ndarray:
            return (
                np.ctypeslib.as_array(getattr(c_frame, field))
                .astype(dtype, copy=True)
                .reshape(8, 8)
            )

        return ToFFrame(
            distance_mm=to_array("distance_mm", np.int16),
            target_status=to_array("target_status", np.uint8),
            target_count=to_array("target_count", np.uint8),
            timestamp_ns=int(c_frame.timestamp_ns),
            streamcount=int(c_frame.streamcount),
        )

    def stop(self) -> None:
        self._require_open()
        if self._started:
            self._check(self._lib.tof_stop(self._device), "stop")
            self._started = False

    def close(self) -> None:
        if self._device:
            dev = self._device
            # Native tof_close() stops ranging if necessary and closes I2C fd.
            self._device = None
            self._started = False
            self._initialized = False
            self._lib.tof_close(dev)

    def __enter__(self) -> VL53L8CX:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
