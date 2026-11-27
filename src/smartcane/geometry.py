"""Orientation and camera coordinate conventions."""

import numpy as np


def quaternion_rotation(quaternion):
    """Return the rotation from IMU coordinates to its world coordinates."""
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Invalid IMU quaternion. Expected finite w, x, y, z.")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_tilt(up):
    pitch = np.degrees(np.arcsin(np.clip(-up[2], -1, 1)))
    roll = np.degrees(np.arctan2(up[0], -up[1]))
    return float(pitch), float(roll)
