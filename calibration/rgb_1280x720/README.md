# RGB camera calibration photos

[Download the 110 original checkerboard photos](https://drive.google.com/file/d/1U0wMXeiQkaM_Pc4oOxf2ZZULLmpctCEn/view?usp=sharing).

Extract them so the first image is `calibration/rgb_1280x720/image_0001.png`.

## Capture your own photos

Start with an empty photo directory. Use the camera and focus setting you plan to use on the cane, at 1280×720, and a checkerboard with **9×6 internal corners**. Run these commands from the repository root on the Cubie. Use `--camera 1` if the camera is at index 1 on Windows.

```sh
uv sync --locked
uv run --locked python scripts/capture_calibration.py --camera /dev/video0
```

Open the URL printed by the script from a browser on the same network. Click **Save photo** or press **S** when the checkerboard is detected. The preview runs at 1 FPS by default. Use `--preview-fps` to change the rate or `--port` if port 8080 is in use. Stop the script with **Ctrl+C** when finished.

The 8×5 grid counts how many saved photos place at least one checkerboard corner in each cell. Yellow outlines show where corners fall in the current frame. Move the board toward cells with few or no samples, and vary its distance and tilt while keeping it in frame. Aim for more than 15 usable views. The script saves full-resolution PNGs without the preview markings and rebuilds the coverage counts when you restart it.

## Fit the camera intrinsics

Measure the width of a printed square and set `SQUARE_SIZE_M` in `scripts/calibration_common.py` in meters. The existing value is 22.45 mm.

```sh
uv run --locked python scripts/calibrate_camera.py
```

The script writes `camera_intrinsics.npz` and `camera_intrinsics.json` to `results/calibration/`. They contain the raw-image camera matrix `K`, distortion coefficients `D`, image size, and overall and per-image reprojection errors. When calibrating with the original photos, pass `--exclude image_0048.png --exclude image_0054.png` to match the original selection.
