# meam5160-group-9-smartcane

AI-powered smart cane for obstacle detection and local environmental mapping, integrating semantic segmentation, ToF ranging, and IMU-based sensor fusion on the Radxa Cubie A7S.

See the [camera calibration dataset guide](calibration/rgb_1280x720/README.md) to download the project's checkerboard photos or capture your own.

Use the [camera–IMU calibration guide](calibration/camera_imu/README.md) to measure the mounted camera and IMU together on the Cubie.

See the [VL53L8CX ranging guide](docs/vl53l8cx.md) to build the ToF driver, read distance frames, and enable the live panel with `--tof`.

## Live CPU segmentation and BEV

On the Cubie, run from the repository root:

```sh
uv run --locked --extra cpu --extra imu python scripts/cane_live_cpu.py --camera /dev/video0 --view 8080
```

Open the printed URL to see the segmentation overlay and ground grid. Add `--bev-image` to check the projection against tape marks on the floor. Use `--log /tmp/smartcane-live.csv` to save frame timing and pose measurements to a new file.

The live script uses the repository's ONNX model, camera intrinsics, and `camera_imu.npz` by default. It interpolates IMU orientation at each camera frame's host timestamp and projects onto a flat floor, assuming the cane tip is on the floor. The grid stays unknown when the frame cannot be paired with nearby IMU samples. The log's IMU timing fields use host arrival times.

## Live A733 NPU segmentation and BEV

To rebuild the `.nb` model from the repository's ONNX export, follow the [WSL conversion guide](docs/npu-wsl.md).

The Cubie A7S uses the A733 VIPLite v2.0 SDK. Copy the files shown below from `viplite-tina/lib/aarch64-none-linux-gnu/v2.0/` in the SDK repository. [SDK files](https://github.com/ZIFENG278/ai-sdk/tree/main/viplite-tina/lib/aarch64-none-linux-gnu/v2.0) · [Radxa setup guide](https://docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/cubie-acuity-usage#download-example-repository)

```text
third_party/viplite/
|-- inc/
|   |-- vip_lite.h
|   `-- vip_lite_common.h
|-- libNBGlinker.so
`-- libVIPhal.so
```

Build the SmartCane library once from the repository root on the Cubie. Rebuild it after changing the C source or the VIPLite SDK.

```sh
VIPLITE_DIR="$PWD/third_party/viplite"

cc -O2 -std=c11 -fPIC -shared -I"$VIPLITE_DIR/inc" \
  src/smartcane/vision/viplite_segmenter.c -L"$VIPLITE_DIR" -lNBGlinker \
  -o "$VIPLITE_DIR/smartcane-viplite.so"
```

Start the live preview from the repository root using the compiled library.

```sh
VIPLITE_DIR="$PWD/third_party/viplite"
NPU_MODEL="$PWD/models/cane_ppliteseg_640x352_pcq_a733.nb"

LD_LIBRARY_PATH="$VIPLITE_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  uv run --locked --extra imu python scripts/cane_live_npu.py \
  --model "$NPU_MODEL" --library "$VIPLITE_DIR/smartcane-viplite.so" \
  --camera /dev/video0 --view 8080
```

The NPU runner expects a three-class PP-LiteSeg model with signed INT8 input and output tensors of shape `1 x 3 x 352 x 640`. Export for A733 with preprocessing outside the graph. The runner reads input quantization parameters from the model and applies the same RGB normalization used by the CPU runner.
