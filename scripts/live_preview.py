"""Background ToF readings and in-memory browser preview frames."""

import time
from collections import deque
from threading import Event, Lock, Thread

import cv2
import numpy as np

from smartcane.sensors.vl53l8cx import VL53L8CX
from smartcane.vision.tof_projection import camera_order

TOF_MAX_FRAME_GAP_NS = 150_000_000


class ToFPreview:
    """Read ToF independently of the frame processing and preview requests."""

    def __init__(self, bus):
        self.bus = bus
        self.lock = Lock()
        self.stopped = Event()
        self.thread = Thread(target=self.read, daemon=True)
        self.frames = deque(maxlen=16)
        self.error = None

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join()

    def read(self):
        try:
            with VL53L8CX(bus=self.bus) as sensor:
                sensor.start()
                while not self.stopped.is_set():
                    frame = sensor.read_frame(timeout_ms=250)
                    if frame is not None:
                        with self.lock:
                            self.frames.append(frame)
        except (OSError, RuntimeError, ValueError) as exc:
            with self.lock:
                self.error = str(exc)

    def frame_at(self, frame_ns):
        with self.lock:
            if self.error is not None:
                return None
            frame = min(
                self.frames,
                key=lambda sample: abs(sample.timestamp_ns - frame_ns),
                default=None,
            )
        if frame is None or abs(frame.timestamp_ns - frame_ns) > TOF_MAX_FRAME_GAP_NS:
            return None
        return frame

    def snapshot(self):
        with self.lock:
            frame = self.frames[-1] if self.frames else None
            return frame, self.error

    def status(self):
        frame, error = self.snapshot()
        if error is not None:
            return {"state": "error", "message": error}
        if frame is None:
            return {"state": "waiting"}
        if time.monotonic_ns() - frame.timestamp_ns > 1_000_000_000:
            return {"state": "stale"}
        tentative = frame.tentative_mask
        distances = np.where(frame.valid_mask | tentative, frame.distance_mm, None)
        return {
            "state": "ready",
            "distances_mm": camera_order(distances).tolist(),
            "tentative": camera_order(tentative).tolist(),
        }


class LivePreview:
    def __init__(self, tof=None):
        self.tof = tof
        self.frames = deque(maxlen=4)
        self.lock = Lock()

    def publish(self, sequence, image, row):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            raise RuntimeError("Could not encode live preview")
        with self.lock:
            self.frames.append((sequence, encoded.tobytes(), row))

    def status(self):
        with self.lock:
            status = {"sequence": None}
            if self.frames:
                sequence, _, row = self.frames[-1]
                status = dict(row, sequence=sequence)
        if self.tof is not None:
            status["tof"] = self.tof.status()
        return status

    def jpeg(self, sequence):
        with self.lock:
            return next(
                (jpeg for number, jpeg, _ in self.frames if number == sequence), None
            )
