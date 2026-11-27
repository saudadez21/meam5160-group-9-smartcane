# VL53L8CX ranging on the Cubie

Use the STSW-IMG040 ULD 2.0.1 package. Copy the files below from its
`VL53L8CX_ULD_API/` directory into `third_party/vl53l8cx/`.
[STMicroelectronics download](https://www.st.com/en/embedded-software/stsw-img040.html)

```text
third_party/vl53l8cx/
|-- LICENSE.txt
|-- inc/
|   |-- vl53l8cx_api.h
|   `-- vl53l8cx_buffers.h
`-- src/
    `-- vl53l8cx_api.c
```

On the Cubie, build and run from the repository root.

```sh
make -C src/smartcane/sensors/vl53l8cx/native
uv run --locked python scripts/read_vl53l8cx.py --bus /dev/i2c-7 --hz 10 --frames 10
```

The runner loads `libvl53l8cx.so` from the native `build/` directory.

The sensor uses the 7-bit I2C address `0x29`. The reader reports one target
per zone in an `8 x 8` grid at 1 to 15 Hz. Distances are in millimeters.
`?` marks tentative returns and `--` marks unknown zones. The timestamp uses
the host monotonic clock when reading a frame completes.

## Sensor orientation and live preview

On this cane, raw row indices increase upward and raw column indices increase
to the left in the camera view. The live grid and projection reverse both
axes to match the image. The driver returns arrays in the original sensor
order.

Add `--view 8080 --tof` to a live BEV command to show the ToF grid.

## Camera alignment preview

Run the standalone preview from the repository root and open the printed URL.

```sh
uv run --locked python scripts/preview_tof_camera.py --camera /dev/video0 --port 8080
```

This preview uses the RGB camera, camera intrinsics, and ToF readings. Filled
cyan points mark valid returns and amber rings mark tentative returns. Labels
are in millimeters. Use `--tof-bus` for a different I2C device and run one
ranging process per sensor.

The overlay uses the ToF frame closest to the camera frame's host timestamp,
within 150 ms. It treats each range as axial depth and assumes parallel
optical axes with the ToF center 3.5 cm to the right of the camera.

For projection, the nominal `45 x 45` degree detection volume is divided
into an `8 x 8` grid on the normalized image plane. Each marker represents
the modeled center of a zone. At 20 cm axial depth, a nominal zone spans
about 2.1 cm on each side. The actual detection volume depends on the target,
lighting, and sensor settings.
[VL53L8CX datasheet, section 2.2](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)

Enable **Show zone outlines** to project each zone's four corners at its
reported depth. Turn it off to return to the center-point view.

## Ground projection preview

Place the cane tip on a flat floor and aim the sensors toward the floor.
Run from the repository root and open the printed URL.

```sh
uv run --locked --extra imu python scripts/preview_tof_ground.py --port 8080
```

The preview uses ToF and IMU readings with the saved camera-IMU mount
calibration. It uses the same metric bounds as the live BEV. The height range
includes reliable returns inside the grid. The floor band defaults to 5 cm
above and below the reference plane and can be adjusted with
`--ground-band-cm`.

Start with a flat floor to check the height range. Keep the cane tip in the
same place while adding and removing an object. For a drop test, use a
known lower surface and compare its measured height with the floor baseline.
