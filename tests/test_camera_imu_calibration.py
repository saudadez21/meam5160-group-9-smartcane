"""Hardware-free checks for the mount fit, PnP geometry, and timestamp pairing."""

import struct
import unittest

import cv2
import numpy as np
from calibrate_camera_imu import fit_extrinsics
from calibration_common import checkerboard_points
from camera_imu_geometry import Intrinsics, measure_board_pose, quaternion_rotation

from smartcane.sensors.bno085 import (
    ImuSample,
    decode_packet,
    match_samples,
    rotation_report,
)


def roll_pitch_quaternion(roll, pitch):
    half_roll, half_pitch = np.radians([roll, pitch]) / 2
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    return np.array([cp * cr, cp * sr, sp * cr, -sp * sr])


class MountFitTest(unittest.TestCase):
    def test_recovers_rotation_and_tip_vector_with_both_axes_varied(self):
        rotation = cv2.Rodrigues(np.array([0.37, -0.24, 0.41]))[0]
        tip = np.array([0.04, -0.12, 0.785])
        tip *= 0.795 / np.linalg.norm(tip)
        pairs = []
        for roll in (-25, -8, 10, 28):
            for pitch in (-35, -22, 3, 25):
                q = roll_pitch_quaternion(roll, pitch)
                up = quaternion_rotation(q)[2]
                pairs.append(
                    (
                        str(len(pairs)),
                        {
                            "q_wxyz": q.tolist(),
                            "up_camera": (rotation.T @ up).tolist(),
                            "height_m": float(up @ tip),
                        },
                    )
                )
        fit = fit_extrinsics(pairs)
        np.testing.assert_allclose(fit["R_ic"], rotation, atol=1e-12)
        np.testing.assert_allclose(fit["tip_to_camera_imu_m"], tip, atol=1e-12)
        self.assertLess(np.max(fit["loo_rotation_deg"]), 1e-5)
        self.assertLess(np.max(np.abs(fit["loo_height_m"])), 1e-12)

    def test_rejects_single_axis_tilt(self):
        pairs = []
        for pitch in np.linspace(-40, 15, 12):
            q = roll_pitch_quaternion(0, pitch)
            up = quaternion_rotation(q)[2]
            pairs.append(
                (
                    str(len(pairs)),
                    {
                        "q_wxyz": q.tolist(),
                        "up_camera": up.tolist(),
                        "height_m": 0.65,
                    },
                )
            )
        with self.assertRaisesRegex(ValueError, "diversity"):
            fit_extrinsics(pairs)

    def test_board_pose_returns_upward_normal_and_optical_center_height(self):
        matrix = np.array([[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
        intrinsics = Intrinsics(matrix, np.zeros(5), "synthetic")
        pitch = np.radians(35.0)
        s, c = np.sin(pitch), np.cos(pitch)
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, -s, -c], [0.0, c, -s]])
        rvec = cv2.Rodrigues(rotation)[0]
        translation = rotation @ [-0.1, 1.0, -0.65]
        corners = cv2.projectPoints(
            checkerboard_points(), rvec, translation, matrix, intrinsics.distortion
        )[0]
        pose = measure_board_pose(corners, intrinsics)
        np.testing.assert_allclose(pose.up_camera, rotation[:, 2], atol=1e-5)
        self.assertAlmostEqual(pose.height_m, 0.65, places=5)


class ImuMatchTest(unittest.TestCase):
    def test_decode_spidev_list(self):
        payload = bytes([0xFB, 0, 0, 0, 0, 0x08, 0, 3, 0])
        payload += struct.pack("<hhhh", 0, 0, 0, 16384)
        wire = list(struct.pack("<HBB", len(payload) + 4, 3, 0) + payload)
        quaternion, status = rotation_report(decode_packet(wire))
        np.testing.assert_array_equal(quaternion, [1, 0, 0, 0])
        self.assertEqual(status, 3)

    def test_average_resolves_quaternion_sign(self):
        center = 1_000_000_000
        q = roll_pitch_quaternion(-15, 30)
        samples = [
            ImuSample(center + i * 20_000_000, (-1) ** i * q, 3) for i in range(-8, 9)
        ]
        match = match_samples(samples, center)
        self.assertAlmostEqual(abs(match.quaternion @ q), 1.0)
        self.assertEqual(match.nearest_ns, 0)
        with self.assertRaisesRegex(ValueError, "around the camera frame"):
            match_samples(samples[:8], center)

    def test_rejects_movement_during_frame_window(self):
        center = 1_000_000_000
        samples = [
            ImuSample(center + i * 20_000_000, roll_pitch_quaternion(i, 0), 3)
            for i in range(-8, 9)
        ]
        with self.assertRaisesRegex(ValueError, "Hold the cane still"):
            match_samples(samples, center)


if __name__ == "__main__":
    unittest.main()
