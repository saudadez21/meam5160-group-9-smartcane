#!/usr/bin/env python3
"""Estimate RGB intrinsics from checkerboard photos with OpenCV.

Run from the repository root with:

uv run python scripts/calibrate_camera.py

Reference: https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html
"""

import argparse
import json

import cv2
import numpy as np
from calibration_common import (
    IMAGE_DIR,
    IMAGE_SIZE,
    IMAGE_SUFFIXES,
    OUTPUT_DIR,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    find_corners,
)

MIN_VIEWS = 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="FILENAME",
        help="Skip this image filename. Repeat for multiple images.",
    )
    args = parser.parse_args()
    excluded_images = set(args.exclude)
    if not IMAGE_DIR.is_dir():
        raise RuntimeError(f"Capture images first: {IMAGE_DIR}")
    paths = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    object_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(
        -1, 2
    )
    object_points *= SQUARE_SIZE_M
    image_points = []
    view_names = []
    for path in paths:
        if path.name in excluded_images:
            print(f"Excluded: {path.name}")
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read: {path}")
        if (frame.shape[1], frame.shape[0]) != IMAGE_SIZE:
            raise ValueError(
                f"Expected {IMAGE_SIZE} images; got {frame.shape[:2]}: {path}"
            )
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if corners is None:
            print(f"Skipped: {path.name}; checkerboard not found")
            continue
        image_points.append(corners)
        view_names.append(path.name)
    if len(image_points) < MIN_VIEWS:
        raise RuntimeError(
            f"Need at least {MIN_VIEWS} usable views in {IMAGE_DIR}. "
            f"Found {len(image_points)}."
        )

    rms, matrix, distortion, *_, errors = cv2.calibrateCameraExtended(
        [object_points] * len(image_points), image_points, IMAGE_SIZE, None, None
    )
    errors = errors.ravel()
    intrinsics = {
        "K": matrix,
        "D": distortion,
        "image_size": IMAGE_SIZE,
        "model": "pinhole",
        "rms": float(rms),
        "square_size_m": SQUARE_SIZE_M,
        "pattern_size": PATTERN_SIZE,
    }
    excluded = sorted(excluded_images)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "camera_intrinsics.npz",
        **intrinsics,
        per_view_rms=errors,
        used_images=view_names,
        excluded_images=np.asarray(excluded, dtype=str),
    )
    data = {
        **intrinsics,
        "K": matrix.tolist(),
        "D": distortion.ravel().tolist(),
        "excluded_images": excluded,
        "per_view_rms": dict(zip(view_names, errors.tolist(), strict=True)),
    }
    (OUTPUT_DIR / "camera_intrinsics.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    worst = errors.argmax()
    print(f"Used {len(view_names)}/{len(paths)} images. Reprojection RMS {rms:.3f} px.")
    print(f"Largest per-image RMS {errors[worst]:.3f} px in {view_names[worst]}.")
    print(f"Wrote camera_intrinsics.npz and camera_intrinsics.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
