# meam5160-group-9-smartcane

AI-powered smart cane for obstacle detection and local environmental mapping, integrating semantic segmentation, ToF ranging, and IMU-based sensor fusion on the Radxa Cubie A7S.

See the [camera calibration dataset guide](calibration/rgb_1280x720/README.md) to download the project's checkerboard photos or capture your own.

Use the [camera–IMU calibration guide](calibration/camera_imu/README.md) to measure the mounted camera and IMU together on the Cubie.

## Live CPU segmentation and BEV

On the Cubie, run from the repository root:

```sh
uv run --locked --extra cpu --extra imu python scripts/cane_live_cpu.py --camera /dev/video0 --view 8080
```

Open the printed URL to see the segmentation overlay and ground grid. Add `--bev-image` to check the projection against tape marks on the floor. Use `--log /tmp/smartcane-live.csv` to save frame timing and pose measurements to a new file.

The live script uses the repository's ONNX model, camera intrinsics, and `camera_imu.npz` by default. It interpolates IMU orientation at each camera frame's host timestamp and projects onto a flat floor, assuming the cane tip is on the floor. The grid stays unknown when the frame cannot be paired with nearby IMU samples. The log's IMU timing fields use host arrival times.
