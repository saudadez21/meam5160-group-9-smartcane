#!/usr/bin/env python3
"""Save raw checkerboard frames. Keys: s saves a detected board; q/Esc exits.

Windows: uv run python scripts/capture_calibration.py --camera 1
"""

import argparse

import cv2
from calibrate_camera import IMAGE_DIR, IMAGE_SIZE, PATTERN_SIZE, find_corners

###############
# Camera

DEVICE = 1
FPS = 25
PIXEL_FORMAT = "MJPG"

###############


def open_camera(camera):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera: {camera}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PIXEL_FORMAT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, FPS)
        print(f"Camera {camera}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):g}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):g}, FPS={cap.get(cv2.CAP_PROP_FPS):g}")
        return cap
    except BaseException:
        cap.release()
        raise


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read camera frame")
    if frame.ndim == 3 and frame.shape[2] == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    if frame.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(f"Unexpected camera frame: {frame.shape}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=DEVICE)
    args = parser.parse_args()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.stem[6:]) for p in IMAGE_DIR.glob("image_*.png") if p.stem[6:].isdigit()
    ]
    index = max(existing, default=0) + 1
    saved = len(existing)
    cap = open_camera(args.camera)
    try:
        print(f"Saving raw images to: {IMAGE_DIR}", flush=True)
        print(
            "Keep the whole board visible; vary its position, distance and tilt.",
            flush=True,
        )
        print("Keys: s saves a detected board; q/Esc exits", flush=True)
        while True:
            frame = read_frame(cap)
            corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners, True)
            label = "Board found" if corners is not None else "Board not found"
            cv2.putText(
                preview,
                f"{label} | saved {saved} | s: save, q: exit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if corners is not None else (0, 0, 255),
                2,
            )
            cv2.imshow("Calibration Capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if corners is None:
                    print("Not saved: checkerboard not found", flush=True)
                    continue
                path = IMAGE_DIR / f"image_{index:04d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Cannot save: {path}")
                index += 1
                saved += 1
                print(f"Saved: {path}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
