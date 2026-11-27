/* Fake ABI-v1 shim. Used ONLY for software testing. */
#include "tof_shim.h"
#include <stdint.h>
#include <string.h>
struct tof_device { int running; int reads; };
static struct tof_device device;
int tof_abi_version(void) { return 1; }
tof_device *tof_open(const char *path) {
    if (!path || strcmp(path, "/dev/i2c-7") != 0) return (tof_device *)0;
    memset(&device, 0, sizeof(device));
    return &device;
}
int tof_initialize(tof_device *d) { return d ? 0 : -1; }
int tof_start(tof_device *d, unsigned hz) {
    if (!d || !hz || hz > 15) return -1;
    d->running = 1;
    return 0;
}
int tof_read(tof_device *d, tof_frame *dst, unsigned ms) {
    (void)ms;
    if (!d || !dst || !d->running) return -1;
    if (d->reads++ > 0) return 0; /* one frame, then a timeout */
    memset(dst, 0, sizeof(*dst));
    dst->resolution = 64;
    dst->streamcount = 7;
    dst->timestamp_ns = 123456789ull;
    dst->distance_mm[0] = 420;
    dst->target_status[0] = 5;
    dst->target_count[0] = 1;
    dst->distance_mm[1] = 250;
    dst->target_status[1] = 6;
    dst->target_count[1] = 1;
    return 1;
}
int tof_stop(tof_device *d) { if (!d) return -1; d->running = 0; return 0; }
void tof_close(tof_device *d) { if (d) d->running = 0; }
const char *tof_last_step(const tof_device *d) { (void)d; return "mock"; }
int tof_last_status(const tof_device *d) { (void)d; return 0; }
