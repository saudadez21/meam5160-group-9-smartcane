#!/usr/bin/env python3
"""Capture raw checkerboard photos using a browser on the local network."""

import argparse
from collections import deque
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np
from calibration_common import IMAGE_DIR, IMAGE_SIZE, IMAGE_SUFFIXES, find_corners
from calibration_web import (
    add_web_args,
    encode_preview,
    open_camera,
    read_frame,
    run_web,
    validate_web_args,
)

GRID_COLS, GRID_ROWS = 8, 5


def covered_cells(corners):
    if corners is None:
        return []
    width, height = IMAGE_SIZE
    points = corners.reshape(-1, 2)
    cols = np.clip((points[:, 0] * GRID_COLS / width).astype(int), 0, GRID_COLS - 1)
    rows = np.clip((points[:, 1] * GRID_ROWS / height).astype(int), 0, GRID_ROWS - 1)
    return np.unique(rows * GRID_COLS + cols).tolist()


@dataclass
class Preview:
    sequence: int
    raw: np.ndarray
    cells: list[int]
    jpeg: bytes
    saved_as: str | None = None


class CaptureSession:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        self.counts = np.zeros(GRID_COLS * GRID_ROWS, dtype=int)
        self.saved = 0
        self.next_index = 1
        self.lock = Lock()
        self.frames = deque(maxlen=4)
        self.sequence = 0
        paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if paths:
            print(f"Reading coverage from {len(paths)} existing photos...", flush=True)
        for path in paths:
            if path.stem.startswith("image_") and path.stem[6:].isdigit():
                self.next_index = max(self.next_index, int(path.stem[6:]) + 1)
            self.saved += 1
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.shape == IMAGE_SIZE[::-1]:
                self.counts[covered_cells(find_corners(gray))] += 1

    def update(self, raw):
        corners = find_corners(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))
        jpeg = encode_preview(raw, corners)
        cells = covered_cells(corners)
        with self.lock:
            self.sequence += 1
            self.frames.append(Preview(self.sequence, raw, cells, jpeg))

    def status(self):
        with self.lock:
            frame = self.frames[-1] if self.frames else None
            return {
                "sequence": frame.sequence if frame else None,
                "active": frame.cells if frame else [],
                "saved_as": frame.saved_as if frame else None,
                "counts": self.counts.tolist(),
                "saved": self.saved,
                "cols": GRID_COLS,
                "rows": GRID_ROWS,
            }

    def jpeg(self, sequence):
        with self.lock:
            return next((p.jpeg for p in self.frames if p.sequence == sequence), None)

    def save(self, sequence):
        with self.lock:
            frame = next((p for p in self.frames if p.sequence == sequence), None)
            if frame is None:
                raise ValueError(
                    "Preview expired. Wait for the next frame and try again."
                )
            if not frame.cells:
                raise ValueError("No complete checkerboard was detected in this frame.")
            if frame.saved_as:
                return frame.saved_as
            ok, encoded = cv2.imencode(".png", frame.raw)
            if not ok:
                raise OSError("Could not encode the capture")
            while True:
                path = self.image_dir / f"image_{self.next_index:04d}.png"
                self.next_index += 1
                try:
                    stream = path.open("xb")
                except FileExistsError:
                    continue
                try:
                    with stream:
                        stream.write(encoded.tobytes())
                except OSError:
                    path.unlink(missing_ok=True)
                    raise
                break
            self.counts[frame.cells] += 1
            self.saved += 1
            frame.saved_as = path.name
            print(f"Saved: {path}", flush=True)
            return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_web_args(parser)
    args = parser.parse_args()
    validate_web_args(parser, args)
    try:
        cap = open_camera(args.camera)
        try:
            session = CaptureSession(IMAGE_DIR)
            run_web(
                session,
                lambda: session.update(read_frame(cap)),
                args.host,
                args.port,
                1 / args.preview_fps,
            )
        finally:
            cap.release()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
