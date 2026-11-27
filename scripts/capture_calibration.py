#!/usr/bin/env python3
"""Capture raw checkerboard photos using a browser on the local network."""

import argparse
import json
import secrets
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np
from calibration_common import (
    IMAGE_DIR,
    IMAGE_SIZE,
    IMAGE_SUFFIXES,
    PATTERN_SIZE,
    find_corners,
)

DEVICE = 1 if sys.platform == "win32" else 0
FPS = 25
PIXEL_FORMAT = "MJPG"
GRID_COLS, GRID_ROWS = 8, 5
PREVIEW_SIZE = (640, 360)


def open_camera(camera):
    backend = {"win32": cv2.CAP_DSHOW, "linux": cv2.CAP_V4L2}.get(
        sys.platform, cv2.CAP_ANY
    )
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera: {camera}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PIXEL_FORMAT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, FPS)
        if backend == cv2.CAP_V4L2:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {camera}: {width:g}x{height:g}, FPS={fps:g}", flush=True)
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


def covered_cells(corners):
    if corners is None:
        return []
    width, height = IMAGE_SIZE
    points = corners.reshape(-1, 2)
    cols = np.clip((points[:, 0] * GRID_COLS / width).astype(int), 0, GRID_COLS - 1)
    rows = np.clip((points[:, 1] * GRID_ROWS / height).astype(int), 0, GRID_ROWS - 1)
    return np.unique(rows * GRID_COLS + cols).tolist()


@dataclass
class Preview:
    sequence: int
    raw: np.ndarray
    cells: list[int]
    jpeg: bytes
    saved_as: str | None = None


class CaptureSession:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        self.counts = np.zeros(GRID_COLS * GRID_ROWS, dtype=int)
        self.saved = 0
        self.next_index = 1
        self.lock = Lock()
        self.frames = deque(maxlen=4)
        self.sequence = 0
        paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if paths:
            print(f"Reading coverage from {len(paths)} existing photos…", flush=True)
        for path in paths:
            if path.stem.startswith("image_") and path.stem[6:].isdigit():
                self.next_index = max(self.next_index, int(path.stem[6:]) + 1)
            self.saved += 1
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.shape == IMAGE_SIZE[::-1]:
                self.counts[covered_cells(find_corners(gray))] += 1

    def update(self, raw):
        corners = find_corners(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))
        preview = cv2.resize(raw, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
        if corners is not None:
            scale = np.array(PREVIEW_SIZE, dtype=np.float32) / np.array(
                IMAGE_SIZE, dtype=np.float32
            )
            cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners * scale, True)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            raise RuntimeError("Could not encode preview")
        cells = covered_cells(corners)
        with self.lock:
            self.sequence += 1
            self.frames.append(Preview(self.sequence, raw, cells, encoded.tobytes()))

    def status(self):
        with self.lock:
            frame = self.frames[-1] if self.frames else None
            return {
                "sequence": frame.sequence if frame else None,
                "active": frame.cells if frame else [],
                "saved_as": frame.saved_as if frame else None,
                "counts": self.counts.tolist(),
                "saved": self.saved,
                "cols": GRID_COLS,
                "rows": GRID_ROWS,
            }

    def jpeg(self, sequence):
        with self.lock:
            return next((p.jpeg for p in self.frames if p.sequence == sequence), None)

    def save(self, sequence):
        with self.lock:
            frame = next((p for p in self.frames if p.sequence == sequence), None)
            if frame is None:
                raise ValueError(
                    "Preview expired. Wait for the next frame and try again."
                )
            if not frame.cells:
                raise ValueError("No complete checkerboard was detected in this frame.")
            if frame.saved_as:
                return frame.saved_as
            ok, encoded = cv2.imencode(".png", frame.raw)
            if not ok:
                raise OSError("Could not encode the capture")
            while True:
                path = self.image_dir / f"image_{self.next_index:04d}.png"
                self.next_index += 1
                try:
                    stream = path.open("xb")
                except FileExistsError:
                    continue
                try:
                    with stream:
                        stream.write(encoded.tobytes())
                except OSError:
                    path.unlink(missing_ok=True)
                    raise
                break
            self.counts[frame.cells] += 1
            self.saved += 1
            frame.saved_as = path.name
            print(f"Saved: {path}", flush=True)
            return path.name


def make_handler(session, prefix, page, interval):
    class CaptureHandler(BaseHTTPRequestHandler):
        timeout = 5

        def log_request(self, code="-", size="-"):
            pass

        def send_content(self, content_type, content, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, data, status=HTTPStatus.OK):
            self.send_content("application/json", json.dumps(data).encode(), status)

        def do_GET(self):
            if self.path == prefix:
                self.send_content("text/html; charset=utf-8", page)
            elif self.path == prefix + "status":
                self.send_json(
                    dict(session.status(), interval_ms=round(interval * 1000))
                )
            elif self.path.startswith(prefix + "preview/"):
                sequence = self.path.removeprefix(prefix + "preview/")
                if not (
                    sequence.isascii() and sequence.isdecimal() and len(sequence) <= 12
                ):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                jpeg = session.jpeg(int(sequence))
                if jpeg is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self.send_content("image/jpeg", jpeg)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != prefix + "save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    self.headers.get_content_type() != "application/json"
                    or not 0 < length <= 128
                ):
                    raise ValueError
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or type(data.get("sequence")) is not int:
                    raise ValueError
            except (ValueError, TimeoutError):
                self.send_json(
                    {"error": "Invalid capture request."}, HTTPStatus.BAD_REQUEST
                )
                return
            try:
                name = session.save(data["sequence"])
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (OSError, cv2.error) as exc:
                print(f"Save failed: {exc}", file=sys.stderr, flush=True)
                self.send_json(
                    {"error": "Could not write the PNG on the board."},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            else:
                self.send_json({"name": name, "state": session.status()})

    return CaptureHandler


def preview_hosts(bind_address):
    if bind_address != "0.0.0.0":
        return [bind_address]
    addresses = {"127.0.0.1"}
    try:
        if sys.platform == "linux":
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            addresses.update(ip for ip in result.stdout.split() if ":" not in ip)
        else:
            addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(addresses)


def run_web(cap, session, host, port, interval):
    prefix = f"/{secrets.token_urlsafe(12)}/"
    page = Path(__file__).with_name("calibration_capture.html").read_bytes()
    with ThreadingHTTPServer(
        (host, port), make_handler(session, prefix, page, interval)
    ) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            print(f"Saving raw images to: {session.image_dir}", flush=True)
            for address in preview_hosts(host):
                print(f"Open http://{address}:{server.server_port}{prefix}", flush=True)
            print("Press Ctrl+C here when finished.", flush=True)
            while True:
                started = time.monotonic()
                session.update(read_frame(cap))
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
        finally:
            server.shutdown()
            thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", default=str(DEVICE), help="camera index or Linux device path"
    )
    parser.add_argument("--host", default="0.0.0.0", help="IPv4 address to listen on")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=1.0,
        help="browser preview rate, 0.2 to 5 FPS",
    )
    args = parser.parse_args()
    if not 0.2 <= args.preview_fps <= 5:
        parser.error("--preview-fps must be between 0.2 and 5")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    try:
        cap = open_camera(camera)
        try:
            run_web(
                cap,
                CaptureSession(IMAGE_DIR),
                args.host,
                args.port,
                1 / args.preview_fps,
            )
        finally:
            cap.release()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
