# Camera and IMU calibration

## Use the project dataset

[Download the 35 camera-IMU calibration pairs](https://drive.google.com/file/d/1GcB_il24mP1QPCoYD_MCAdXm8UoRWeFw/view?usp=sharing).

Extract the ZIP so each pair lands under `calibration/camera_imu/pair_<timestamp>/`.

## Capture on the Cubie

Use the final camera and BNO085 mount. Measure the printed checkerboard square and verify `SQUARE_SIZE_M` in `scripts/calibration_common.py`. Generate the camera intrinsics for that camera, focus and resolution before collecting joint poses.

From the repository root, install the Linux IMU dependencies and start the browser preview.

```sh
uv sync --locked --extra imu
uv run --locked --extra imu python scripts/capture_camera_imu.py --camera /dev/video0
```

Open the printed URL on the same network. The default BNO085 wiring is `/dev/spidev1.0`, INT `/dev/gpiochip0:312`, RST `/dev/gpiochip0:313`, and WAKE `/dev/gpiochip1:5` on PCB V2. Use `--spi-device`, `--int-line`, `--rst-line`, and `--wake-line` if your wiring differs. Close any other process using the camera or IMU before starting.

1. Lay the checkerboard flat on the same floor as the cane tip. Keep the entire board in view and the cane tip on that floor for each pose.
2. Hold the cane still until the preview says **Ready**, then click **Save pose** or press **S**. The preview defaults to 1 FPS. Each capture saves the raw PNG, camera ground-plane estimate, frame time and nearby timestamped IMU samples in a `pair_…` directory here.
3. Collect at least 10 distinct poses. Vary forward tilt and tilt to both sides, including combinations of the two. The grid shows which pitch and roll ranges have samples. The diversity score checks whether the ground normals span enough directions for a three-dimensional fit.

The camera is drained continuously while the browser preview runs at low frame rate. The pair uses host arrival timestamps, with a short stationary IMU window on both sides of the frame timestamp. This setup measures the fixed mount geometry. Dynamic camera-to-IMU timing during a swing is a separate test.

## Fit and verify

```sh
uv run --locked python scripts/calibrate_camera_imu.py
```

The fitter reports angle and height residuals, including a leave-one-out check for each pose. Capture more side tilt if the diversity score is low. If one pose has a large held-out error, review its saved frame and recapture it with the tip on the floor and the board fully visible. Move the suspect pair out of the capture directory before fitting again.

A passing fit writes `results/calibration/camera_imu.npz` with `R_ic` (camera to IMU rotation), `tip_to_camera_imu_m` (vector from the tip to the camera optical center in IMU coordinates), and quality metrics. While the tip is on the floor, the dot product of that vector with the upward direction in IMU coordinates gives camera height. The file also records which intrinsics were used so mismatched calibration files can be detected.

The result included in this repository was fitted for our assembled cane and the included camera intrinsics.

On our assembled cane, the straight-line distance from the tip's floor contact point to the camera's optical center is 79.5 cm. For a different cane or mount, measure that distance and set `TIP_DISTANCE_M` in [`scripts/camera_imu_geometry.py`](../../scripts/camera_imu_geometry.py) in meters. The fitter estimates the vector from the checkerboard heights and IMU orientations. It compares the fitted length with the measured distance and prints a warning if they differ by more than 4 cm.

Collect a fresh set of poses after changing the mount or camera intrinsics.
