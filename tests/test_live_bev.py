"""Check live frame pairing and the camera-relative ground projection."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from cane_live import process_frame
from live_preview import LivePreview, ToFPreview
from preview_tof_ground import grid_background, render_ground_points

from smartcane.sensors.bno085 import (
    FrameOrientation,
    ImuSample,
    interpolate_orientation,
)
from smartcane.sensors.vl53l8cx import ToFFrame
from smartcane.vision.bev import UNKNOWN, GroundGrid, GroundPose, MountCalibration
from smartcane.vision.segmentation_common import MODEL_SIZE, preprocess
from smartcane.vision.tof_projection import project_tof, tof_ground_points


class FramePairingTest(unittest.TestCase):
    def test_interpolates_short_arc_at_frame_time(self):
        identity = np.array([1.0, 0, 0, 0])
        angle = np.radians(30)
        rotated = np.array([np.cos(angle), np.sin(angle), 0, 0])
        samples = [ImuSample(0, identity, 3), ImuSample(20_000_000, -rotated, 3)]
        match = interpolate_orientation(samples, 5_000_000)
        np.testing.assert_allclose(
            match.quaternion, [np.cos(angle / 4), np.sin(angle / 4), 0, 0]
        )
        self.assertEqual(match.nearest_ns, 5_000_000)
        self.assertEqual(match.gap_ns, 20_000_000)

    def test_rejects_extrapolation_and_long_gaps(self):
        q = np.array([1.0, 0, 0, 0])
        samples = [ImuSample(0, q, 3), ImuSample(80_000_000, q, 3)]
        with self.assertRaisesRegex(ValueError, "bracket"):
            interpolate_orientation(samples, 90_000_000)
        with self.assertRaisesRegex(ValueError, "gap"):
            interpolate_orientation(samples, 40_000_000)


class ToFPreviewTest(unittest.TestCase):
    def test_camera_view_preserves_the_raw_frame(self):
        distance = np.zeros((8, 8), dtype=np.int16)
        status = np.zeros((8, 8), dtype=np.uint8)
        count = np.ones((8, 8), dtype=np.uint8)
        distance[7, 7], status[7, 7] = 400, 5
        distance[0, 0], status[0, 0] = 230, 6
        distance[4, 5], status[4, 5] = 650, 1
        frame = ToFFrame(distance, status, count, timestamp_ns=1, streamcount=0)
        tof = ToFPreview("/dev/i2c-7")
        tof.frames.append(frame)
        with patch("live_preview.time.monotonic_ns", return_value=2):
            preview = LivePreview(tof).status()
        self.assertIsNone(preview["sequence"])
        self.assertEqual(preview["tof"]["distances_mm"][0][0], 400)
        self.assertEqual(preview["tof"]["distances_mm"][7][7], 230)
        self.assertIsNone(preview["tof"]["distances_mm"][3][2])
        self.assertFalse(preview["tof"]["tentative"][0][0])
        self.assertTrue(preview["tof"]["tentative"][7][7])
        self.assertEqual(frame.distance_mm[0, 0], 230)
        self.assertEqual(frame.target_status[7, 7], 5)

    def test_stale_frames_are_hidden(self):
        tof = ToFPreview("/dev/i2c-7")
        self.assertEqual(tof.status(), {"state": "waiting"})
        tof.frames.append(Mock(timestamp_ns=0))
        with patch("live_preview.time.monotonic_ns", return_value=1_000_000_001):
            self.assertEqual(tof.status(), {"state": "stale"})

    def test_uses_nearest_tof_frame_within_camera_time_window(self):
        tof = ToFPreview("/dev/i2c-7")
        nearest = Mock(timestamp_ns=100_000_000)
        latest = Mock(timestamp_ns=300_000_000)
        tof.frames.extend((nearest, latest))
        self.assertIs(tof.frame_at(170_000_000), nearest)
        self.assertIsNone(tof.frame_at(500_000_000))

    def test_projects_raw_corners_with_positive_camera_x_offset(self):
        distance = np.full((8, 8), 1000, dtype=np.int16)
        status = np.zeros((8, 8), dtype=np.uint8)
        count = np.ones((8, 8), dtype=np.uint8)
        status[7, 7], status[0, 0] = 5, 6
        frame = ToFFrame(distance, status, count, timestamp_ns=1, streamcount=0)
        matrix = np.array([[400.0, 0, 320], [0, 400.0, 176], [0, 0, 1]])
        pixels, depths, tentative = project_tof(frame, matrix, np.zeros(5))
        centers = pixels[:, 0]
        np.testing.assert_array_equal(depths, [1000, 1000])
        np.testing.assert_array_equal(tentative, [False, True])
        np.testing.assert_allclose(centers.mean(axis=0), [334, 176])
        self.assertTrue((centers[0] < (320, 176)).all())
        self.assertTrue((centers[1] > (320, 176)).all())
        np.testing.assert_allclose(pixels[:, 1:].mean(axis=1), centers)

    def test_outline_shifts_with_depth_for_parallel_sensor_axes(self):
        status = np.zeros((8, 8), dtype=np.uint8)
        status[3, 4] = 5
        count = np.ones((8, 8), dtype=np.uint8)
        matrix = np.array([[400.0, 0, 320], [0, 400.0, 176], [0, 0, 1]])
        outlines = []
        for mm in (200, 400):
            depth = np.full((8, 8), mm, dtype=np.int16)
            frame = ToFFrame(depth, status, count, timestamp_ns=0, streamcount=0)
            pixels, _, _ = project_tof(frame, matrix, np.zeros(5))
            outlines.append(pixels[0, 1:])
        displacement = outlines[0] - outlines[1]
        np.testing.assert_allclose(displacement, [[35.0, 0.0]] * 4)

    def test_ground_coordinates_keep_signed_height(self):
        distances = np.full((8, 8), 1000, dtype=np.int16)
        status = np.zeros((8, 8), dtype=np.uint8)
        status[3, 3] = 5
        count = np.ones((8, 8), dtype=np.uint8)
        frame = ToFFrame(distances, status, count, timestamp_ns=0, streamcount=0)
        camera_ground = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        pose = GroundPose(camera_ground, np.array([0, 0.7, 0]), 0, 0, 0.7)
        xyz, tentative = tof_ground_points(frame, pose)
        ray = np.tan(np.radians(22.5)) / 8
        np.testing.assert_allclose(xyz, [[0.035 + ray, 1.0, 0.7 - ray]])
        np.testing.assert_array_equal(tentative, [False])

    def test_ground_preview_separates_tentative_and_outside_returns(self):
        image = grid_background()
        points = np.array(
            [
                [0.0, 0.5, 0.02],
                [0.1, 0.5, 0.16],
                [0.2, 0.5, -0.18],
                [0.0, 0.7, 0.2],
                [2.0, 0.7, 0.3],
            ]
        )
        tentative = np.array([False, False, False, True, False])
        result = render_ground_points(image, points, tentative, 0.05)
        self.assertEqual(
            result,
            {
                "ground": 1,
                "above": 1,
                "below": 1,
                "tentative": 1,
                "outside": 1,
                "reliable_heights_cm": [2.0, 16.0, -18.0],
            },
        )


class LiveProjectionTest(unittest.TestCase):
    def setUp(self):
        pitch = np.radians(35)
        s, c = np.sin(pitch), np.cos(pitch)
        camera_to_imu = np.array([[1.0, 0, 0], [0, -s, c], [0, -c, -s]])
        self.mount = MountCalibration(camera_to_imu, np.array([0.4, 0, 0.65]))
        self.matrix = np.array([[700.0, 0, 640], [0, 700, 360], [0, 0, 1]])
        self.grid = GroundGrid(self.matrix, np.zeros(5), (1280, 720))
        self.q = np.array([1.0, 0, 0, 0])

    def test_height_and_pitch_from_mount_and_quaternion(self):
        pose = self.mount.ground_pose(self.q)
        self.assertAlmostEqual(pose.height_m, 0.65)
        self.assertAlmostEqual(pose.pitch_deg, 35)
        self.assertAlmostEqual(pose.roll_deg, 0)
        lookup = self.grid.lookup(pose, MODEL_SIZE)
        labels = np.zeros(MODEL_SIZE[::-1], dtype=np.uint8)
        projected = lookup.sample(labels, UNKNOWN)
        self.assertTrue(np.any(projected == 0))
        self.assertTrue(np.any(projected == UNKNOWN))
        self.assertTrue(np.all(lookup.x < MODEL_SIZE[0]))
        self.assertTrue(np.all(lookup.y < MODEL_SIZE[1]))

    def test_camera_roll_rotates_the_ground_axes(self):
        radians = np.radians(10)
        q = np.array([np.cos(radians), 0, np.sin(radians), 0])
        pose = self.mount.ground_pose(q)
        up_imu = np.array([-np.sin(2 * radians), 0, np.cos(2 * radians)])
        np.testing.assert_allclose(
            pose.rotation[:, 2], self.mount.camera_to_imu.T @ up_imu
        )
        self.assertAlmostEqual(
            pose.height_m, float(up_imu @ self.mount.tip_to_camera_imu_m)
        )
        self.assertNotAlmostEqual(pose.roll_deg, 0)

    def test_frame_pose_is_chosen_before_inference(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        imu, segmenter = Mock(), Mock()
        imu.orientation_at.return_value = FrameOrientation(
            self.q, 5_000_000, 20_000_000
        )

        def predict(raw):
            imu.orientation_at.assert_called_once_with(123)
            return raw[:352, :640], np.zeros(MODEL_SIZE[::-1], dtype=np.uint8)

        segmenter.predict.side_effect = predict
        row, _, _, _ = process_frame(
            123, frame, imu, self.mount, segmenter, self.grid, 0.5
        )
        self.assertEqual(row["decision"], "go straight")
        self.assertEqual(row["imu_nearest_ms"], 5)
        imu.orientation_at.side_effect = ValueError("No samples")
        segmenter.predict.side_effect = lambda raw: (
            raw[:352, :640],
            np.zeros(MODEL_SIZE[::-1], dtype=np.uint8),
        )
        row, _, _, lookup = process_frame(
            456, frame, imu, self.mount, segmenter, self.grid, 0.5
        )
        self.assertIsNone(lookup)
        self.assertEqual(row["decision"], "unknown")

    def test_preprocess_uses_rgb_and_training_normalization(self):
        image = np.full((720, 1280, 3), [255, 0, 0], dtype=np.uint8)
        _, tensor = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 352, 640))
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [-1, -1, 1])

    def test_upward_camera_reports_pose_but_skips_ground_projection(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        imu, mount, segmenter = Mock(), Mock(), Mock()
        imu.orientation_at.return_value = FrameOrientation(self.q, 0, 0)
        mount.ground_pose.return_value = GroundPose(np.eye(3), np.zeros(3), -28, 0, 0.1)
        segmenter.predict.return_value = (
            frame[:352, :640],
            np.zeros(MODEL_SIZE[::-1], dtype=np.uint8),
        )
        row, _, _, lookup = process_frame(
            123, frame, imu, mount, segmenter, self.grid, 0.5
        )
        self.assertEqual(row["pitch_deg"], -28)
        self.assertEqual(row["decision"], "unknown")
        self.assertIsNone(lookup)


if __name__ == "__main__":
    unittest.main()
