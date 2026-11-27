/* Linux i2c-dev adapter for ST VL53L8CX ULD 2.0.1. */
#define _POSIX_C_SOURCE 200809L
#include "platform.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

/* Conservative size: ST firmware write requests can be tens of KB. */
#define BLOCK_BYTES 240u

int vl53l8cx_linux_open(VL53L8CX_Platform *p, const char *path, uint8_t addr7)
{
    if (!p || !path || addr7 < 0x08 || addr7 > 0x77) {
        errno = EINVAL;
        return -1;
    }
    p->fd = -1;
    p->address = (uint16_t)(addr7 << 1); /* ST 8-bit address convention */
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    unsigned long funcs = 0;
    if (ioctl(fd, I2C_FUNCS, &funcs) < 0) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    if (!(funcs & I2C_FUNC_I2C)) {
        close(fd);
        errno = ENOTSUP;
        return -1;
    }
    p->fd = fd;
    return 0;
}

void vl53l8cx_linux_close(VL53L8CX_Platform *p)
{
    if (p && p->fd >= 0) {
        close(p->fd);
        p->fd = -1;
    }
}

static uint8_t transact(VL53L8CX_Platform *p, uint16_t reg,
                        uint8_t *bytes, uint16_t count, int read_mode)
{
    if (!p || p->fd < 0 || !bytes || count == 0 || count > BLOCK_BYTES)
        return 255;
    uint8_t address_bytes[2] = {(uint8_t)(reg >> 8), (uint8_t)reg};
    uint8_t out[2 + BLOCK_BYTES];
    struct i2c_msg msg[2];
    unsigned n = read_mode ? 2u : 1u;
    uint16_t slave = (uint16_t)(p->address >> 1);

    if (read_mode) {
        msg[0] = (struct i2c_msg){.addr=slave, .len=2, .flags=0, .buf=address_bytes};
        msg[1] = (struct i2c_msg){.addr=slave, .len=count, .flags=I2C_M_RD, .buf=bytes};
    } else {
        memcpy(out, address_bytes, 2);
        memcpy(out + 2, bytes, count);
        msg[0] = (struct i2c_msg){.addr=slave, .len=(uint16_t)(count + 2), .flags=0, .buf=out};
    }
    struct i2c_rdwr_ioctl_data request = {.msgs=msg, .nmsgs=n};
    int result = ioctl(p->fd, I2C_RDWR, &request);
    if (result != (int)n) {
        fprintf(stderr, "I2C %s reg 0x%04X failed: %s\n", read_mode ? "read" : "write",
                reg, result < 0 ? strerror(errno) : "short transaction");
        return 255;
    }
    return 0;
}

uint8_t VL53L8CX_RdMulti(VL53L8CX_Platform *p, uint16_t reg,
                         uint8_t *data, uint32_t count)
{
    if (!p || (!data && count) || (uint64_t)reg + count > 65536u) return 255;
    for (uint32_t off = 0; off < count;) {
        uint16_t n = (uint16_t)((count - off > BLOCK_BYTES) ? BLOCK_BYTES : count - off);
        if (transact(p, (uint16_t)(reg + off), data + off, n, 1)) return 255;
        off += n;
    }
    return 0;
}
uint8_t VL53L8CX_WrMulti(VL53L8CX_Platform *p, uint16_t reg,
                         uint8_t *data, uint32_t count)
{
    if (!p || (!data && count) || (uint64_t)reg + count > 65536u) return 255;
    for (uint32_t off = 0; off < count;) {
        uint16_t n = (uint16_t)((count - off > BLOCK_BYTES) ? BLOCK_BYTES : count - off);
        if (transact(p, (uint16_t)(reg + off), data + off, n, 0)) return 255;
        off += n;
    }
    return 0;
}
uint8_t VL53L8CX_RdByte(VL53L8CX_Platform *p, uint16_t r, uint8_t *d)
{
    return VL53L8CX_RdMulti(p, r, d, 1);
}
uint8_t VL53L8CX_WrByte(VL53L8CX_Platform *p, uint16_t r, uint8_t d)
{
    return VL53L8CX_WrMulti(p, r, &d, 1);
}
uint8_t VL53L8CX_WaitMs(VL53L8CX_Platform *p, uint32_t ms)
{
    (void)p;
    struct timespec ts = {(time_t)(ms / 1000u), (long)((ms % 1000u) * 1000000u)};
    while (nanosleep(&ts, &ts) < 0) if (errno != EINTR) return 255;
    return 0;
}
void VL53L8CX_SwapBuffer(uint8_t *b, uint16_t size)
{
    if (!b || (size % 4u)) return;
    for (uint16_t i = 0; i < size; i += 4) {
        uint8_t temp = b[i]; b[i] = b[i + 3]; b[i + 3] = temp;
        temp = b[i + 1]; b[i + 1] = b[i + 2]; b[i + 2] = temp;
    }
}
uint8_t VL53L8CX_Reset_Sensor(VL53L8CX_Platform *p)
{
    (void)p;
    return 255;
}
