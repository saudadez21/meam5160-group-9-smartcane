"""Shared calibration settings and checkerboard detection."""

from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = _ROOT / "calibration/rgb_1280x720"
OUTPUT_DIR = _ROOT / "results/calibration"
IMAGE_SIZE = (1280, 720)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PATTERN_SIZE = (9, 6)  # Internal corners in columns and rows.
SQUARE_SIZE_M = 0.02245  # Side length of one square on the printed board.


def checkerboard_points():
    cols, rows = PATTERN_SIZE
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2)
    points *= SQUARE_SIZE_M
    return points


def find_corners(gray):
    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
