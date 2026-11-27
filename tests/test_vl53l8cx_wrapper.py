"""No-sensor unit tests for ABI, value semantics, status filtering, and timeouts."""

import ctypes as C
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smartcane.sensors.vl53l8cx import VL53L8CX, ToFFrame
from smartcane.sensors.vl53l8cx.vl53l8cx import _CFrame


class TestFrame(unittest.TestCase):
    def test_abi_layout(self):
        self.assertEqual(C.sizeof(_CFrame), 272)
        self.assertEqual(_CFrame.timestamp_ns.offset, 264)

    def test_status_filter(self):
        distance = np.full((8, 8), 500, dtype=np.int16)
        status = np.zeros((8, 8), dtype=np.uint8)
        count = np.ones((8, 8), dtype=np.uint8)
        status[0, 0] = 5
        status[0, 1] = 6
        status[0, 2] = 9
        status[0, 3] = 5
        count[0, 3] = 0
        status[0, 4] = 5
        distance[0, 4] = -1
        f = ToFFrame(distance, status, count, timestamp_ns=123, streamcount=3)
        self.assertEqual(int(f.valid_mask.sum()), 1)
        self.assertEqual(int(f.tentative_mask.sum()), 2)
        self.assertEqual(int(np.isfinite(f.distances_for_mapping()).sum()), 1)
        self.assertEqual(
            int(np.isfinite(f.distances_for_mapping(include_tentative=True)).sum()), 3
        )
        self.assertTrue(np.isnan(f.distances_for_mapping()[2, 2]))

    def test_no_library(self):
        with TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            VL53L8CX(lib_path=Path(directory) / "libvl53l8cx.so")


if __name__ == "__main__":
    unittest.main()
