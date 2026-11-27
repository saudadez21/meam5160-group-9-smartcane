CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=c11
ULD_DIR ?= ../../../../../third_party/vl53l8cx
CPPFLAGS += -I. -I$(ULD_DIR)/inc

.PHONY: all clean
all: build/libvl53l8cx.so

build:
	mkdir -p $@

# STSW-IMG040 2.0.1 is missing a semicolon in the firmware checksum error branch.
build/vl53l8cx_api.c: $(ULD_DIR)/src/vl53l8cx_api.c Makefile | build
	sed 's/^\([[:space:]]*goto exit\)[[:space:]]*$$/\1;/' "$<" > "$@"

build/libvl53l8cx.so: build/vl53l8cx_api.c platform_linux.c platform.h \
    tof_shim.c tof_shim.h $(ULD_DIR)/inc/vl53l8cx_api.h \
    $(ULD_DIR)/inc/vl53l8cx_buffers.h Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -fPIC -shared \
	    platform_linux.c tof_shim.c build/vl53l8cx_api.c $(LDLIBS) -o $@

clean:
	rm -rf build
