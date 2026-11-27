#!/usr/bin/env python3
"""Check ToF ground projection using the cane's IMU and mount calibration."""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from calibration_web import add_web_args, preview_server, validate_web_args
from camera_imu_geometry import EXTRINSICS_PATH, INTRINSICS_PATH, load_intrinsics
from cubie_io import add_imu_args, imu_stream
from live_preview import LivePreview, ToFPreview

from smartcane.vision.bev import GRID_FAR_M, GRID_NEAR_M, GRID_WIDTH_M, MountCalibration
from smartcane.vision.tof_projection import tof_ground_points

PIXELS_PER_M = 200
GRID_SIZE = (
    round(GRID_WIDTH_M * PIXELS_PER_M),
    round((GRID_FAR_M - GRID_NEAR_M) * PIXELS_PER_M),
)
GROUND = (174, 207, 97)
ABOVE = (109, 137, 255)
BELOW = (255, 167, 110)
TENTATIVE = (72, 184, 244)


def grid_background():
    width, height = GRID_SIZE
    image = np.full((height, width, 3), (29, 24, 19), dtype=np.uint8)
    for distance in (0.5, 1.0, 1.5, 2.0, 2.5):
        y = round((GRID_FAR_M - distance) * PIXELS_PER_M)
        cv2.line(image, (0, y), (width - 1, y), (85, 77, 69), 1)
        cv2.putText(
            image,
            f"{distance:g} m",
            (6, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (207, 203, 199),
            1,
            cv2.LINE_AA,
        )
    for lateral in (-0.5, 0.0, 0.5):
        x = round((lateral + GRID_WIDTH_M / 2) * PIXELS_PER_M)
        color = (105, 98, 91) if lateral == 0 else (65, 58, 51)
        cv2.line(image, (x, 0), (x, height - 1), color, 1)
    return image


def render_ground_points(image, points, tentative, band_m):
    """Draw visible zone centers and report height evidence for the preview."""
    pixels = (
        np.column_stack((points[:, 0] + GRID_WIDTH_M / 2, GRID_FAR_M - points[:, 1]))
        * PIXELS_PER_M
    )
    inside = ((pixels >= 0) & (pixels < GRID_SIZE)).all(axis=1)
    centers = np.floor(pixels[inside]).astype(int)
    heights = points[inside, 2]
    tentative = tentative[inside]
    ground = (np.abs(heights) <= band_m) & ~tentative
    above = (heights > band_m) & ~tentative
    below = (heights < -band_m) & ~tentative
    for mask, color, thickness in (
        (tentative, TENTATIVE, 1),
        (ground, GROUND, -1),
        (above, ABOVE, -1),
        (below, BELOW, -1),
    ):
        for center in centers[mask]:
            cv2.circle(
                image,
                tuple(map(int, center)),
                4,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return {
        "ground": int(ground.sum()),
        "above": int(above.sum()),
        "below": int(below.sum()),
        "tentative": int(tentative.sum()),
        "outside": int((~inside).sum()),
        "reliable_heights_cm": (100 * heights[~tentative]).round(1).tolist(),
    }


def ground_frame(frame, imu, mount, band_m, image):
    orientation = imu.orientation_at(frame.timestamp_ns)
    pose = mount.ground_pose(orientation.quaternion)
    if pose.pitch_deg <= 0:
        raise ValueError("Point the sensors toward the floor.")
    points, tentative = tof_ground_points(frame, pose)
    state = render_ground_points(image, points, tentative, band_m)
    return dict(
        state,
        pitch_deg=pose.pitch_deg,
        camera_height_m=pose.height_m,
        imu_gap_ms=orientation.gap_ns / 1e6,
        tof_timestamp_ns=frame.timestamp_ns,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_web_args(parser, camera=False)
    add_imu_args(parser)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--mount", type=Path, default=EXTRINSICS_PATH)
    parser.add_argument("--tof-bus", default="/dev/i2c-7", help="ToF I2C device")
    parser.add_argument("--ground-band-cm", type=float, default=5.0)
    args = parser.parse_args()
    validate_web_args(parser, args)
    if not math.isfinite(args.ground_band_cm) or args.ground_band_cm <= 0:
        parser.error("--ground-band-cm must be a positive finite number")
    interval = 1 / args.preview_fps
    band_m = args.ground_band_cm / 100
    try:
        intrinsics = load_intrinsics(args.intrinsics)
        mount = MountCalibration.load(args.mount, intrinsics.digest)
        page = Path(__file__).with_name("tof_ground_preview.html").read_bytes()
        background = grid_background()
        with imu_stream(args) as imu, ToFPreview(args.tof_bus) as tof:
            viewer = LivePreview()
            with preview_server(
                viewer, args.host, args.port, page, interval, allow_save=False
            ):
                sequence = 0
                while True:
                    started = time.monotonic()
                    image = background.copy()
                    frame, error = tof.snapshot()
                    if error is not None:
                        state = {"message": f"ToF unavailable. {error}"}
                    elif frame is None:
                        state = {"message": "Waiting for ToF data."}
                    elif time.monotonic_ns() - frame.timestamp_ns > 1_000_000_000:
                        state = {"message": "No recent ToF data."}
                    else:
                        try:
                            state = ground_frame(frame, imu, mount, band_m, image)
                        except ValueError as exc:
                            state = {"message": str(exc)}
                    viewer.publish(
                        sequence, image, dict(state, band_cm=args.ground_band_cm)
                    )
                    sequence += 1
                    time.sleep(max(0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except (KeyError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
