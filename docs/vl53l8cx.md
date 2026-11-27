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

With the sensor mounted on this cane, face the sensor from in front of the
cane. Raw rows run from bottom to top and raw columns run from right to left.
The live panel reverses both axes so it matches that view. The driver returns
arrays in the original sensor order.

Add `--view 8080 --tof` to either live BEV command to show the ToF panel.
Use `--tof-bus` if the sensor is on a different I2C device. Run one ranging
process per sensor. The panel displays the latest ToF frame independently of
the bird's-eye view.
