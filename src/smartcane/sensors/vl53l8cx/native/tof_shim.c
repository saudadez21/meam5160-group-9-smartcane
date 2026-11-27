#define _POSIX_C_SOURCE 200809L
#include "tof_shim.h"
#include "vl53l8cx_api.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct tof_device {
    VL53L8CX_Configuration cfg;
    VL53L8CX_ResultsData result;
    char last_step[64];
    int last_status;
    int initialized;
    int ranging;
};

static int set_error(tof_device *d, const char *step, int status) {
    if (d) {
        (void)snprintf(d->last_step, sizeof(d->last_step), "%s", step);
        d->last_status = status;
    }
    return -1;
}
static int good(tof_device *d) {
    d->last_step[0] = '\0';
    d->last_status = 0;
    return 0;
}
static int msleep(unsigned ms) {
    struct timespec t = {(time_t)(ms/1000u), (long)(ms%1000u)*1000000l};
    while (nanosleep(&t, &t) < 0) if (errno != EINTR) return -1;
    return 0;
}
static uint64_t monotonic_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) < 0) return 0;
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

tof_device *tof_open(const char *path) {
    if (!path) return NULL;
    tof_device *d = (tof_device*)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->cfg.platform.fd = -1;
    if (vl53l8cx_linux_open(&d->cfg.platform, path, 0x29) != 0) {
        free(d);
        return NULL;
    }
    good(d);
    return d;
}
int tof_initialize(tof_device *d) {
    if (!d || d->ranging) return -1;
    if (d->initialized) return 0;
    uint8_t alive = 0;
    uint8_t s = vl53l8cx_is_alive(&d->cfg, &alive);
    if (s) return set_error(d, "is_alive (I2C device ID)", s);
    if (!alive) return set_error(d, "unexpected device ID", 255);
    /* The ULD software reboot + 84KiB firmware load runs here. */
    s = vl53l8cx_init(&d->cfg);
    if (s) return set_error(d, "vl53l8cx_init (firmware/config)", s);
    d->initialized = 1;
    return good(d);
}
int tof_start(tof_device *d, unsigned hz) {
    if (!d || !d->initialized || d->ranging || hz < 1u || hz > 15u) return -1;
    uint8_t s;
    s = vl53l8cx_set_resolution(&d->cfg, VL53L8CX_RESOLUTION_8X8);
    if (s) return set_error(d, "set_resolution(8x8)", s);
    s = vl53l8cx_set_ranging_frequency_hz(&d->cfg, (uint8_t)hz);
    if (s) return set_error(d, "set_ranging_frequency_hz", s);
    s = vl53l8cx_set_ranging_mode(&d->cfg, VL53L8CX_RANGING_MODE_CONTINUOUS);
    if (s) return set_error(d, "set_ranging_mode(CONTINUOUS)", s);
    s = vl53l8cx_start_ranging(&d->cfg);
    if (s) return set_error(d, "start_ranging", s);
    d->ranging = 1;
    return good(d);
}
int tof_read(tof_device *d, tof_frame *dst, unsigned timeout_ms) {
    if (!d || !dst || !d->ranging) return -1;
    uint64_t start = monotonic_ns();
    const uint64_t deadline = start + (uint64_t)timeout_ms * 1000000ull;
    for (;;) {
        uint8_t ready = 0;
        uint8_t s = vl53l8cx_check_data_ready(&d->cfg, &ready);
        if (s) return set_error(d, "check_data_ready", s);
        if (ready) {
            memset(&d->result, 0, sizeof(d->result));
            s = vl53l8cx_get_ranging_data(&d->cfg, &d->result);
            if (s) return set_error(d, "get_ranging_data", s);
            memset(dst, 0, sizeof(*dst));
            memcpy(dst->distance_mm, d->result.distance_mm, sizeof(dst->distance_mm));
            memcpy(dst->target_status, d->result.target_status, sizeof(dst->target_status));
            memcpy(dst->target_count, d->result.nb_target_detected, sizeof(dst->target_count));
            dst->resolution = 64;
            dst->streamcount = d->cfg.streamcount;
            dst->timestamp_ns = monotonic_ns();
            good(d);
            return 1;
        }
        if (monotonic_ns() >= deadline) return 0;
        if (msleep(10u) < 0) return set_error(d, "sleep/poll", 255);
    }
}
int tof_stop(tof_device *d) {
    if (!d) return -1;
    if (!d->ranging) return 0;
    uint8_t s = vl53l8cx_stop_ranging(&d->cfg);
    d->ranging = 0;
    if (s) return set_error(d, "stop_ranging", s);
    return good(d);
}
void tof_close(tof_device *d) {
    if (!d) return;
    if (d->ranging) (void)tof_stop(d);
    vl53l8cx_linux_close(&d->cfg.platform);
    free(d);
}
const char *tof_last_step(const tof_device *d) { return d ? d->last_step : "invalid handle"; }
int tof_last_status(const tof_device *d) { return d ? d->last_status : -1; }
int tof_abi_version(void) { return 1; }
