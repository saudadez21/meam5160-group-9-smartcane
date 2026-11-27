#!/usr/bin/env python3
"""Capture paired camera and IMU poses with a checkerboard flat on the floor."""

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import cv2
import numpy as np
from calibration_common import find_corners
from calibration_web import (
    add_web_args,
    encode_preview,
    run_web,
    validate_web_args,
)
from camera_imu_geometry import (
    INTRINSICS_PATH,
    MIN_DIVERSITY,
    MIN_PAIRS,
    PAIR_DIR,
    load_intrinsics,
    load_pairs,
    measure_board_pose,
    pose_cell,
    pose_diversity,
)
from cubie_io import CameraFrames, add_imu_args, imu_stream

from smartcane.geometry import camera_tilt, quaternion_rotation


@dataclass
class Preview:
    sequence: int
    raw: np.ndarray
    jpeg: bytes
    active: list[int]
    message: str
    record: dict | None
    saved_as: str | None = None


class JointCaptureSession:
    def __init__(self, output, intrinsics, camera, imu):
        self.image_dir = output
        self.intrinsics = intrinsics
        self.camera = camera
        self.imu = imu
        self.frames = deque(maxlen=4)
        self.sequence = 0
        self.lock = Lock()
        self.up_directions = []
        self.imu_up_directions = []
        self.counts = np.zeros(9, dtype=int)
        for _, pair in load_pairs(output, intrinsics.digest):
            self.count_pair(pair)
        output.mkdir(parents=True, exist_ok=True)

    def count_pair(self, record):
        up = np.asarray(record["up_camera"])
        self.up_directions.append(up)
        self.imu_up_directions.append(quaternion_rotation(record["q_wxyz"])[2])
        self.counts[pose_cell(up)] += 1

    def update(self):
        frame_ns, raw = self.camera.latest()
        corners = find_corners(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))
        active = []
        record = None
        message = "Show the whole checkerboard on the floor."
        if corners is not None:
            try:
                pose = measure_board_pose(corners, self.intrinsics)
                active = [pose_cell(pose.up_camera)]
                pitch, roll = camera_tilt(pose.up_camera)
                matched = self.imu.match(frame_ns)
                message = (
                    f"Ready | pitch {pitch:.1f} deg | roll {roll:+.1f} deg | "
                    f"height {pose.height_m * 100:.1f} cm | "
                    f"IMU spread {matched.spread_deg:.2f} deg"
                )
                record = {
                    "frame_host_ns": frame_ns,
                    "intrinsics_sha256": self.intrinsics.digest,
                    "q_wxyz": matched.quaternion.tolist(),
                    "imu_spread_deg": matched.spread_deg,
                    "nearest_imu_ms": matched.nearest_ns / 1e6,
                    "imu_samples": [
                        {
                            "host_ns": sample.host_ns,
                            "q_wxyz": sample.quaternion.tolist(),
                            "status": sample.status,
                        }
                        for sample in matched.samples
                    ],
                    "up_camera": pose.up_camera.tolist(),
                    "height_m": pose.height_m,
                    "pitch_deg": pitch,
                    "roll_deg": roll,
                    "reprojection_rms_px": pose.rms_px,
                }
            except ValueError as exc:
                message = str(exc)
        jpeg = encode_preview(raw, corners)
        with self.lock:
            self.sequence += 1
            self.frames.append(
                Preview(self.sequence, raw, jpeg, active, message, record)
            )

    def status(self):
        with self.lock:
            frame = self.frames[-1] if self.frames else None
            diversity = min(
                pose_diversity(self.up_directions),
                pose_diversity(self.imu_up_directions),
            )
            return {
                "mode": "camera_imu",
                "sequence": frame.sequence if frame else None,
                "active": frame.active if frame else [],
                "saved_as": frame.saved_as if frame else None,
                "can_save": bool(frame and frame.record and not frame.saved_as),
                "message": frame.message
                if frame
                else "Waiting for a camera and IMU frame.",
                "counts": self.counts.tolist(),
                "saved": len(self.up_directions),
                "minimum_pairs": MIN_PAIRS,
                "diversity": round(diversity, 3),
                "minimum_diversity": MIN_DIVERSITY,
                "cols": 3,
                "rows": 3,
            }

    def jpeg(self, sequence):
        with self.lock:
            return next(
                (frame.jpeg for frame in self.frames if frame.sequence == sequence),
                None,
            )

    def save(self, sequence):
        with self.lock:
            frame = next(
                (frame for frame in self.frames if frame.sequence == sequence), None
            )
            if frame is None:
                raise ValueError(
                    "Preview expired. Wait for the next frame and try again."
                )
            if frame.record is None:
                raise ValueError(frame.message)
            if frame.saved_as:
                return frame.saved_as
            name = f"pair_{time.time_ns()}"
            with TemporaryDirectory(dir=self.image_dir, prefix=".capture-") as temp:
                pending = Path(temp)
                if not cv2.imwrite(str(pending / "frame.png"), frame.raw):
                    raise OSError("Could not write the capture image")
                (pending / "pose.json").write_text(
                    json.dumps(frame.record, indent=2) + "\n", encoding="utf-8"
                )
                pending.rename(self.image_dir / name)
            self.count_pair(frame.record)
            frame.saved_as = name
            print(f"Saved {self.image_dir / name}", flush=True)
            return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_web_args(parser)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--output", type=Path, default=PAIR_DIR)
    add_imu_args(parser)
    args = parser.parse_args()
    validate_web_args(parser, args)
    if sys.platform != "linux":
        parser.error("The BNO085 SPI capture tool requires Linux on the Cubie")
    try:
        intrinsics = load_intrinsics(args.intrinsics)
        with (
            CameraFrames(args.camera) as camera,
            imu_stream(args) as imu,
        ):
            session = JointCaptureSession(args.output, intrinsics, camera, imu)
            run_web(session, session.update, args.host, args.port, 1 / args.preview_fps)
    except KeyboardInterrupt:
        pass
    except (KeyError, RuntimeError, OSError, ValueError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
