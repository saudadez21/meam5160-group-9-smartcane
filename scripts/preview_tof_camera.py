#!/usr/bin/env python3
"""Preview ToF points on RGB for checking the sensor alignment."""

import argparse
import time
from pathlib import Path

import cv2
from calibration_common import IMAGE_SIZE
from calibration_web import (
    PREVIEW_SIZE,
    add_web_args,
    preview_server,
    validate_web_args,
)
from camera_imu_geometry import INTRINSICS_PATH, load_intrinsics
from cubie_io import CameraFrames
from live_preview import LivePreview, ToFPreview

from smartcane.vision.tof_projection import draw_tof_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_web_args(parser)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--tof-bus", default="/dev/i2c-7", help="ToF I2C device")
    args = parser.parse_args()
    validate_web_args(parser, args)
    interval = 1 / args.preview_fps
    try:
        intrinsics = load_intrinsics(args.intrinsics)
        page = Path(__file__).with_name("tof_camera_preview.html").read_bytes()
        with CameraFrames(args.camera) as camera, ToFPreview(args.tof_bus) as tof:
            viewer = LivePreview(tof)
            with preview_server(
                viewer, args.host, args.port, page, interval, allow_save=False
            ):
                frame_ns, sequence = -1, 0
                while True:
                    started = time.monotonic()
                    frame_ns, raw = camera.latest(after_ns=frame_ns)
                    image = cv2.resize(raw, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
                    frame = tof.frame_at(frame_ns)
                    points, gap_ms, outlines = None, None, []
                    if frame is not None:
                        points, outlines = draw_tof_points(
                            image,
                            frame,
                            intrinsics.matrix,
                            intrinsics.distortion,
                            IMAGE_SIZE,
                        )
                        gap_ms = abs(frame.timestamp_ns - frame_ns) / 1e6
                    viewer.publish(
                        sequence,
                        image,
                        {"points": points, "tof_gap_ms": gap_ms, "outlines": outlines},
                    )
                    sequence += 1
                    time.sleep(max(0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except (KeyError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
