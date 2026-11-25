#!/usr/bin/env python3
"""Estimate RGB intrinsics from checkerboard photos with OpenCV.

Windows: uv run python scripts/calibrate_camera.py

Reference: https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html
"""

from pathlib import Path
import json

import cv2
import numpy as np

###############
# Images and checkerboard

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "calibration/rgb_1280x720"
OUTPUT_DIR = ROOT / "results/calibration"
IMAGE_SIZE = (1280, 720)
PATTERN_SIZE = (9, 6)  # Internal corners: columns, rows.
# SQUARE_SIZE_M = 0.025
SQUARE_SIZE_M = 0.02245  # Estimated printed square size (m).
MIN_VIEWS = 15
EXCLUDED_IMAGES = {"image_0048.png", "image_0054.png"}

###############
def find_corners(gray):
    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)


def main():
    if not IMAGE_DIR.is_dir():
        raise RuntimeError(f"Capture images first: {IMAGE_DIR}")
    paths = sorted(
        p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    object_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(
        -1, 2
    )
    object_points *= SQUARE_SIZE_M
    image_points = []
    used_paths = []
    for path in paths:
        if path.name in EXCLUDED_IMAGES:
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
        used_paths.append(path)
    if len(image_points) < MIN_VIEWS:
        raise RuntimeError(
            f"Need at least {MIN_VIEWS} usable views; found {len(image_points)} in {IMAGE_DIR}"
        )

    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [object_points] * len(image_points), image_points, IMAGE_SIZE, None, None
    )

    report = [
        "Model: pinhole (5 coefficients)",
        f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        f"Board: {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} internal corners; {SQUARE_SIZE_M * 1000:g} mm squares",
        f"Images used: {len(image_points)} / {len(paths)}",
        f"Excluded: {sorted(EXCLUDED_IMAGES)}",
        f"Reprojection RMS: {rms:.6f} px",
        "Camera matrix K:",
        np.array2string(matrix, precision=6, suppress_small=True),
        "Distortion (k1, k2, p1, p2, k3):",
        np.array2string(distortion.ravel(), precision=8, suppress_small=True),
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for path, corners, rvec, tvec in zip(used_paths, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
        residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        errors.append(error)
        report.append(f"{path.name}: {error:.6f} px")

    new_k, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, IMAGE_SIZE, 0.0, IMAGE_SIZE)
    report.extend(["Rectified K (alpha=0, no crop):", np.array2string(new_k, precision=6), f"Valid ROI: {roi}"])
    map1, map2 = cv2.initUndistortRectifyMap(matrix, distortion, None, new_k, IMAGE_SIZE, cv2.CV_32FC1)
    valid = (np.isfinite(map1) & np.isfinite(map2) & (map1 >= 0) & (map1 < IMAGE_SIZE[0]-1) & (map2 >= 0) & (map2 < IMAGE_SIZE[1]-1))
    cv2.imwrite(str(OUTPUT_DIR / "valid_pixel_mask.png"), valid.astype(np.uint8)*255)
    np.savez(OUTPUT_DIR / "camera_intrinsics.npz", K=matrix, D=distortion,
             K_new=new_k, image_size=np.array(IMAGE_SIZE), model="pinhole",
             rms=rms, roi=np.array(roi), square_size_m=SQUARE_SIZE_M,
             pattern_size=np.array(PATTERN_SIZE), per_view_rms=np.array(errors), used_images=np.array([p.name for p in used_paths]), excluded_images=np.array(sorted(EXCLUDED_IMAGES)))
    data = dict(K=matrix.tolist(), D=distortion.ravel().tolist(), K_new=new_k.tolist(),
                image_size=list(IMAGE_SIZE), model="pinhole", rms=float(rms),
                roi=[int(v) for v in roi], alpha=0.0, square_size_m=SQUARE_SIZE_M,
                pattern_size=list(PATTERN_SIZE), excluded_images=sorted(EXCLUDED_IMAGES), per_view_rms=dict(zip([p.name for p in used_paths], errors)))
    (OUTPUT_DIR / "camera_intrinsics.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    original = cv2.imread(str(used_paths[0]))
    rectified = cv2.undistort(original, matrix, distortion, None, new_k)
    preview = np.hstack((original, rectified))
    if not cv2.imwrite(str(OUTPUT_DIR / "undistortion_preview.png"), preview):
        raise RuntimeError("Could not save preview")
    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
