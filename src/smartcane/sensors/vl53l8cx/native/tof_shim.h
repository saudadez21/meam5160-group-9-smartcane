/* VL53L8CX Linux interface used by the Python ctypes driver. */
#ifndef SMARTCANE_TOF_SHIM_H
#define SMARTCANE_TOF_SHIM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tof_device tof_device;
/* Fixed 8x8, single target per zone, row-major ST zone order.
 * The physical image orientation must be validated on the assembled cane.
 */
typedef struct tof_frame {
    int16_t distance_mm[64];
    uint8_t target_status[64];
    uint8_t target_count[64];
    uint8_t streamcount;
    uint8_t resolution; /* 64 */
    uint8_t reserved[6];
    uint64_t timestamp_ns; /* CLOCK_MONOTONIC, host read completion */
} tof_frame;

/* On error, NULL. Linux 7-bit device address is fixed to 0x29. */
tof_device *tof_open(const char *i2c_path);
/* 0 = success; non-zero = failure. Call tof_last_step/status for details. */
int tof_initialize(tof_device *dev);
int tof_start(tof_device *dev, unsigned frequency_hz); /* 1..15 Hz at 8x8 */
/* 1 = new frame, 0 = timeout (not an error), -1 = error */
int tof_read(tof_device *dev, tof_frame *dst, unsigned timeout_ms);
int tof_stop(tof_device *dev);
void tof_close(tof_device *dev);
const char *tof_last_step(const tof_device *dev);
int tof_last_status(const tof_device *dev);
int tof_abi_version(void); /* currently 1 */
#ifdef __cplusplus
}
#endif
#endif
