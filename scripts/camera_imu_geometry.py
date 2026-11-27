"""Ground-plane observations and coordinate conventions for camera-IMU calibration."""

import hashlib
import io
import json
from dataclasses import dataclass

import cv2
import numpy as np
from calibration_common import (
    IMAGE_DIR,
    IMAGE_SIZE,
    OUTPUT_DIR,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    checkerboard_points,
)

from smartcane.geometry import camera_tilt

PAIR_DIR = IMAGE_DIR.parent / "camera_imu"
INTRINSICS_PATH = OUTPUT_DIR / "camera_intrinsics.npz"
EXTRINSICS_PATH = OUTPUT_DIR / "camera_imu.npz"
TIP_DISTANCE_M = 0.795
MIN_PAIRS = 10
MIN_DIVERSITY = 0.12
MAX_BOARD_RMS_PX = 0.8


@dataclass(frozen=True)
class Intrinsics:
    matrix: np.ndarray
    distortion: np.ndarray
    digest: str


def load_intrinsics(path):
    contents = path.read_bytes()
    with np.load(io.BytesIO(contents), allow_pickle=False) as data:
        if tuple(data["image_size"]) != IMAGE_SIZE or data["model"].item() != "pinhole":
            raise ValueError(
                f"Expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pinhole intrinsics"
            )
        if tuple(data["pattern_size"]) != PATTERN_SIZE or not np.isclose(
            data["square_size_m"], SQUARE_SIZE_M
        ):
            raise ValueError(
                "Checkerboard configuration differs from the intrinsics file"
            )
        return Intrinsics(data["K"], data["D"], hashlib.sha256(contents).hexdigest())


@dataclass(frozen=True)
class BoardPose:
    up_camera: np.ndarray
    height_m: float
    rms_px: float


def measure_board_pose(corners, intrinsics):
    points = checkerboard_points()
    ok, rvec, tvec = cv2.solvePnP(
        points,
        corners,
        intrinsics.matrix,
        intrinsics.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("Cannot estimate the checkerboard pose")
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec.reshape(3)
    if np.any((points @ rotation.T + translation)[:, 2] <= 0):
        raise ValueError("Checkerboard points projected behind the camera")
    up = rotation[:, 2].copy()
    if up @ translation > 0:
        up *= -1
    projected = cv2.projectPoints(
        points, rvec, tvec, intrinsics.matrix, intrinsics.distortion
    )[0]
    residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    height = float(-up @ translation)
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Checkerboard pose gives an invalid camera height")
    if not np.isfinite(rms) or rms > MAX_BOARD_RMS_PX:
        raise ValueError(
            f"Checkerboard fit RMS {rms:.2f} px. Reposition or hold still."
        )
    return BoardPose(up, height, rms)


def pose_cell(up):
    pitch, roll = camera_tilt(up)
    row = np.searchsorted([25, 40], pitch)
    col = np.searchsorted([-8, 8], roll)
    return int(3 * row + col)


def load_pairs(directory, digest):
    pairs = []
    for path in sorted(directory.glob("pair_*/pose.json")):
        try:
            pair = json.loads(path.read_text(encoding="utf-8"))
            if pair["intrinsics_sha256"] != digest:
                raise ValueError(
                    "intrinsics file differs from the one used for capture"
                )
            pairs.append((path.parent.name, pair))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    return pairs


def pose_diversity(up_directions):
    if len(up_directions) < 3:
        return 0.0
    singular_values = np.linalg.svd(up_directions, compute_uv=False)
    return float(singular_values[-1] / singular_values[0])
