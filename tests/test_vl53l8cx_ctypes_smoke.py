"""Hardware-free ctypes integration test using a compiled fake C ABI-v1 .so."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from smartcane.sensors.vl53l8cx import VL53L8CX, VL53L8CXError


@unittest.skipUnless(sys.platform == "linux", "Requires a Linux shared library")
class TestFakeNativeShim(unittest.TestCase):
    def test_fake_native_round_trip(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler not installed")
        test_dir = Path(__file__).resolve().parent
        native = test_dir.parent / "src/smartcane/sensors/vl53l8cx/native"
        with tempfile.TemporaryDirectory() as tmp:
            so = Path(tmp) / "libvl53l8cx_mock.so"
            subprocess.run(
                [
                    cc,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fPIC",
                    "-shared",
                    "-I",
                    str(native),
                    str(test_dir / "fake_tof_shim.c"),
                    "-o",
                    str(so),
                ],
                check=True,
            )
            with VL53L8CX(bus=7, lib_path=so) as tof:
                tof.start(resolution=64, frequency_hz=10)
                frame = tof.read_frame(timeout_ms=200)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.distance_mm.shape, (8, 8))
                self.assertEqual(frame.distance_mm.dtype.name, "int16")
                self.assertEqual(frame.target_status.dtype.name, "uint8")
                self.assertEqual(frame.timestamp_ns, 123456789)
                self.assertEqual(frame.streamcount, 7)
                self.assertEqual(frame.distance_mm[0, 0], 420)
                self.assertTrue(frame.valid_mask[0, 0])
                self.assertTrue(frame.tentative_mask[0, 1])
                self.assertEqual(frame.distances_for_mapping()[0, 0], 420.0)
                self.assertIsNone(tof.read_frame(timeout_ms=50))
                tof.stop()
            with self.assertRaisesRegex(VL53L8CXError, "closed"):
                tof.read_frame()


if __name__ == "__main__":
    unittest.main()
