#!/usr/bin/env python3
"""Read 8x8 VL53L8CX distances on the Cubie."""

import argparse
import sys
from pathlib import Path

from smartcane.sensors.vl53l8cx import VL53L8CX


def print_frame(number, frame):
    print(f"Frame {number} | stream {frame.streamcount} | host ns {frame.timestamp_ns}")
    valid, tentative = frame.valid_mask, frame.tentative_mask
    for distances, reliable, maybe in zip(
        frame.distance_mm, valid, tentative, strict=True
    ):
        cells = []
        for distance, accepted, uncertain in zip(
            distances, reliable, maybe, strict=True
        ):
            if accepted:
                cells.append(f"{distance:5d} ")
            elif uncertain:
                cells.append(f"{distance:5d}?")
            else:
                cells.append("    --")
        print(" ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus", default="/dev/i2c-7")
    parser.add_argument("--lib", type=Path, help="path to libvl53l8cx.so")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=1500)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if not 1 <= args.hz <= 15:
        parser.error("--hz must be between 1 and 15")
    if not 0 <= args.timeout_ms <= 60000:
        parser.error("--timeout-ms must be between 0 and 60000")

    received = 0
    try:
        with VL53L8CX(bus=args.bus, lib_path=args.lib) as sensor:
            print(f"Starting 8x8 ranging at {args.hz} Hz on {args.bus}", flush=True)
            sensor.start(frequency_hz=args.hz)
            for number in range(1, args.frames + 1):
                frame = sensor.read_frame(timeout_ms=args.timeout_ms)
                if frame is None:
                    print(f"Frame {number} | timeout")
                    continue
                received += 1
                print_frame(number, frame)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Received {received}/{args.frames} frames")
    return 0 if received else 1


if __name__ == "__main__":
    raise SystemExit(main())
