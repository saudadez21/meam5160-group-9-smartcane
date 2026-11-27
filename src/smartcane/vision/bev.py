"""Project segmentation labels onto a camera-relative ground plane."""

from dataclasses import dataclass

import cv2
import numpy as np

from smartcane.geometry import camera_tilt, quaternion_rotation

UNKNOWN = 255
GRID_CELL_M = 0.05
GRID_NEAR_M = 0.3
GRID_FAR_M = 2.5
GRID_WIDTH_M = 2.0


@dataclass(frozen=True)
class GroundPose:
    rotation: np.ndarray
    translation: np.ndarray
    pitch_deg: float
    roll_deg: float
    height_m: float

    def camera_to_ground(self, points):
        """Transform camera points into the ground grid's metric coordinates."""
        return (points - self.translation) @ self.rotation


@dataclass(frozen=True)
class MountCalibration:
    camera_to_imu: np.ndarray
    tip_to_camera_imu_m: np.ndarray

    @classmethod
    def load(cls, path, intrinsics_digest):
        with np.load(path, allow_pickle=False) as data:
            if data["intrinsics_sha256"].item() != intrinsics_digest:
                raise ValueError(
                    "Camera-IMU calibration does not match the camera intrinsics"
                )
            rotation = data["R_ic"]
            tip_vector = data["tip_to_camera_imu_m"]
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0)
            or tip_vector.shape != (3,)
            or not np.all(np.isfinite(tip_vector))
            or np.linalg.norm(tip_vector) <= 0
        ):
            raise ValueError("Invalid camera-IMU mount geometry")
        return cls(rotation, tip_vector)

    def ground_pose(self, quaternion):
        # The cane tip is assumed to be on the floor.
        up_imu = quaternion_rotation(quaternion)[2]
        up_camera = self.camera_to_imu.T @ up_imu
        height = float(up_imu @ self.tip_to_camera_imu_m)
        if height <= 0.05:
            raise ValueError("Camera height is too low for ground projection")
        forward = np.array([0.0, 0.0, 1.0]) - up_camera[2] * up_camera
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 0.05:
            raise ValueError("Optical axis is nearly vertical")
        forward /= forward_norm
        right = np.cross(forward, up_camera)
        rotation = np.column_stack((right, forward, up_camera))
        pitch, roll = camera_tilt(up_camera)
        return GroundPose(rotation, -height * up_camera, pitch, roll, height)


@dataclass(frozen=True)
class GridSamples:
    shape: tuple[int, int]
    cells: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def sample(self, image, fill):
        result = np.full((*self.shape, *image.shape[2:]), fill, dtype=image.dtype)
        flat = result.reshape((-1, *image.shape[2:]))
        flat[self.cells] = image[self.y, self.x]
        return result


class GroundGrid:
    def __init__(self, matrix, distortion, raw_size):
        self.matrix, self.distortion, self.raw_size = matrix, distortion, raw_size
        cols = round(GRID_WIDTH_M / GRID_CELL_M)
        rows = round((GRID_FAR_M - GRID_NEAR_M) / GRID_CELL_M)
        self.x = -GRID_WIDTH_M / 2 + (np.arange(cols) + 0.5) * GRID_CELL_M
        self.y = GRID_FAR_M - (np.arange(rows) + 0.5) * GRID_CELL_M
        x, y = np.meshgrid(self.x, self.y)
        self.shape = x.shape
        self.points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        # Reject rays past the first fold in the radial distortion polynomial.
        k1, k2, _, _, k3 = distortion.ravel()
        roots = np.roots([7 * k3, 5 * k2, 3 * k1, 1])
        positive_roots = [
            root.real for root in roots if abs(root.imag) < 1e-9 and root.real > 0
        ]
        self.max_radius_sq = min(positive_roots, default=np.inf)

    def lookup(self, pose, sample_size):
        camera = self.points @ pose.rotation.T + pose.translation
        cells = np.flatnonzero(camera[:, 2] > 0.05)
        rays = camera[cells, :2] / camera[cells, 2:3]
        cells = cells[np.sum(rays**2, axis=1) < self.max_radius_sq]
        if not cells.size:
            return GridSamples(self.shape, cells, cells, cells)
        pixels = cv2.projectPoints(
            camera[cells], np.zeros(3), np.zeros(3), self.matrix, self.distortion
        )[0].reshape(-1, 2)
        raw_size = np.asarray(self.raw_size)
        valid = ((pixels >= 0) & (pixels <= raw_size - 1)).all(axis=1)
        scaled = np.floor(
            (pixels[valid] + 0.5) * np.asarray(sample_size) / raw_size
        ).astype(int)
        return GridSamples(self.shape, cells[valid], scaled[:, 0], scaled[:, 1])

    def lane_scores(self, labels):
        near = (self.y >= 0.5) & (self.y <= 1.5)
        scores = []
        for left, right in ((-0.75, -0.25), (-0.25, 0.25), (0.25, 0.75)):
            lane = labels[np.ix_(near, (self.x >= left) & (self.x < right))]
            known = lane[lane != UNKNOWN]
            scores.append(float(np.mean(known == 0)) if known.size else None)
        return scores


def lane_decision(scores, threshold):
    left, center, right = scores
    if left is None or center is None or right is None:
        return "unknown"
    if center >= threshold:
        return "go straight"
    if max(left, right) >= threshold:
        return "go left" if left >= right else "go right"
    return "stop"
