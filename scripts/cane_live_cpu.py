#!/usr/bin/env python3
"""Run CPU semantic segmentation and frame-aligned BEV on the Cubie."""

import argparse
import csv
import math
import sys
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from calibration_common import IMAGE_SIZE
from calibration_web import preview_server
from camera_imu_geometry import EXTRINSICS_PATH, INTRINSICS_PATH, load_intrinsics
from cubie_io import DEVICE, CameraFrames, add_imu_args, imu_stream

from smartcane.vision.bev import (
    GRID_CELL_M,
    GRID_FAR_M,
    UNKNOWN,
    GroundGrid,
    MountCalibration,
    lane_decision,
)
from smartcane.vision.segmentation_cpu import COLORS, MODEL_SIZE, CpuSegmenter, overlay

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/cane_ppliteseg_640x352.onnx"


def process_frame(frame_ns, raw, imu, mount, segmenter, grid, threshold):
    row = {
        "frame_host_ns": frame_ns,
        "imu_nearest_ms": None,
        "imu_gap_ms": None,
        "pitch_deg": None,
        "roll_deg": None,
        "height_m": None,
        "visible_fraction": 0.0,
        "left": None,
        "center": None,
        "right": None,
        "decision": "unknown",
        "pose_status": "matched",
    }
    lookup = None
    try:
        orientation = imu.orientation_at(frame_ns)
        row["imu_nearest_ms"] = round(orientation.nearest_ns / 1e6, 2)
        row["imu_gap_ms"] = round(orientation.gap_ns / 1e6, 2)
        pose = mount.ground_pose(orientation.quaternion)
        row.update(
            pitch_deg=round(pose.pitch_deg, 2),
            roll_deg=round(pose.roll_deg, 2),
            height_m=round(pose.height_m, 3),
        )
        if pose.pitch_deg <= 0:
            raise ValueError("Camera is level or angled upward")
        lookup = grid.lookup(pose, MODEL_SIZE)
        row["visible_fraction"] = round(len(lookup.cells) / grid.points.shape[0], 3)
    except ValueError as exc:
        row["pose_status"] = str(exc)

    started = time.monotonic_ns()
    resized, labels = segmenter.predict(raw)
    row["infer_ms"] = round((time.monotonic_ns() - started) / 1e6, 1)
    if lookup is not None:
        scores = grid.lane_scores(lookup.sample(labels, UNKNOWN))
        row["left"], row["center"], row["right"] = [
            round(score, 3) if score is not None else None for score in scores
        ]
        row["decision"] = lane_decision(scores, threshold)
    row["frame_age_ms"] = round((time.monotonic_ns() - frame_ns) / 1e6, 1)
    return row, resized, labels, lookup


def bev_panel(grid, lookup, resized, labels, real_image):
    if lookup is None:
        colors = np.full((*grid.shape, 3), 50, dtype=np.uint8)
    elif real_image:
        colors = lookup.sample(resized, 50)
    else:
        projected = lookup.sample(labels, UNKNOWN)
        colors = np.full((*grid.shape, 3), 50, dtype=np.uint8)
        known = projected != UNKNOWN
        colors[known] = COLORS[projected[known]]
    panel = cv2.resize(colors, (320, MODEL_SIZE[1]), interpolation=cv2.INTER_NEAREST)
    height, width = panel.shape[:2]
    for distance in (0.5, 1.0, 1.5, 2.0, 2.5):
        row = round((GRID_FAR_M - distance) / GRID_CELL_M * height / grid.shape[0])
        cv2.line(panel, (0, row), (width - 1, row), (220, 220, 220), 1)
        cv2.putText(
            panel,
            f"{distance:g} m",
            (4, max(14, row - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    cv2.line(panel, (width // 2, 0), (width // 2, height - 1), (0, 255, 255), 1)
    return panel


class LivePreview:
    def __init__(self):
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
            if not self.frames:
                return {"sequence": None}
            sequence, _, row = self.frames[-1]
            return dict(row, sequence=sequence)

    def jpeg(self, sequence):
        with self.lock:
            return next(
                (jpeg for number, jpeg, _ in self.frames if number == sequence), None
            )


def print_frame(number, row):
    line = (
        f"{number:05d} | infer {row['infer_ms']:.0f} ms | "
        f"age {row['frame_age_ms']:.0f} ms"
    )
    if row["pose_status"] != "matched":
        print(f"{line} | BEV unknown: {row['pose_status']}", flush=True)
        return
    scores = " / ".join(
        f"{row[key]:.2f}" if row[key] is not None else "-"
        for key in ("left", "center", "right")
    )
    print(
        f"{line} | IMU {row['imu_nearest_ms']:.1f} ms | "
        f"pitch {row['pitch_deg']:+.1f} deg roll {row['roll_deg']:+.1f} deg "
        f"h {row['height_m']:.2f} m | L/C/R {scores} | {row['decision']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", default=str(DEVICE), help="camera index or Linux device path"
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--mount", type=Path, default=EXTRINSICS_PATH)
    parser.add_argument(
        "--threads", type=int, default=4, help="ONNX Runtime CPU threads"
    )
    parser.add_argument("--seconds", type=float, default=0, help="0 runs until Ctrl+C")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="walkable fraction required per lane",
    )
    parser.add_argument(
        "--view", type=int, metavar="PORT", help="serve a browser preview"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="IPv4 address for the browser preview"
    )
    parser.add_argument(
        "--bev-image",
        action="store_true",
        help="display projected RGB for checking floor marks",
    )
    parser.add_argument("--log", type=Path, help="write metrics to this new CSV file")
    add_imu_args(parser)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Live BEV requires the Cubie Linux SPI IMU")
    if args.threads < 1 or not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--threads must be positive and --seconds must be nonnegative")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be greater than 0 and at most 1")
    if args.view is not None and not 0 <= args.view <= 65535:
        parser.error("--view must be a port between 0 and 65535")
    try:
        intrinsics = load_intrinsics(args.intrinsics)
        mount = MountCalibration.load(args.mount, intrinsics.digest)
        grid = GroundGrid(intrinsics.matrix, intrinsics.distortion, IMAGE_SIZE)
        segmenter = CpuSegmenter(args.model, args.threads)
        with ExitStack() as stack:
            imu = stack.enter_context(imu_stream(args))
            camera = stack.enter_context(CameraFrames(args.camera))
            stream = (
                stack.enter_context(args.log.open("x", encoding="utf-8", newline=""))
                if args.log
                else None
            )
            writer = None
            viewer = None
            if args.view is not None:
                viewer = LivePreview()
                page = Path(__file__).with_name("live_bev.html").read_bytes()
                stack.enter_context(
                    preview_server(
                        viewer, args.host, args.view, page, 1.0, allow_save=False
                    )
                )
            print(
                "CPU segmentation ready. BEV assumes the cane tip is on the floor.",
                flush=True,
            )
            start, frame_ns, count = time.monotonic(), -1, 0
            while not args.seconds or time.monotonic() - start < args.seconds:
                frame_ns, raw = camera.latest(after_ns=frame_ns)
                row, resized, labels, lookup = process_frame(
                    frame_ns, raw, imu, mount, segmenter, grid, args.threshold
                )
                if stream is not None:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                if viewer is not None:
                    bev = bev_panel(grid, lookup, resized, labels, args.bev_image)
                    viewer.publish(
                        count, np.hstack((overlay(resized, labels), bev)), row
                    )
                print_frame(count, row)
                count += 1
    except KeyboardInterrupt:
        pass
    except (KeyError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
