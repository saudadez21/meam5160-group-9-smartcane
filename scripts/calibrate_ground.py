#!/usr/bin/env python3
"""Estimate a fixed camera pose from a checkerboard lying flat on the ground."""
import argparse
import hashlib
import json
import cv2
import numpy as np
from calibrate_camera import OUTPUT_DIR, IMAGE_SIZE, PATTERN_SIZE, SQUARE_SIZE_M, find_corners
from capture_calibration import DEVICE, open_camera, read_frame

INTRINSICS = OUTPUT_DIR / 'camera_intrinsics.npz'
GROUND = OUTPUT_DIR / 'ground_extrinsics.npz'


def load_intrinsics():
    with np.load(INTRINSICS, allow_pickle=False) as p:
        if tuple(p['image_size']) != IMAGE_SIZE or str(p['model'].item()) != 'pinhole':
            raise ValueError('Expected 1280x720 pinhole intrinsics')
        if tuple(p['pattern_size']) != PATTERN_SIZE or not np.isclose(p['square_size_m'], SQUARE_SIZE_M):
            raise ValueError('Checkerboard configuration differs from intrinsics')
        return p['K'], p['D'], p['K_new']


def intrinsics_hash():
    return hashlib.sha256(INTRINSICS.read_bytes()).hexdigest()


def estimate_pose(corners, K, D):
    obj = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
    obj[:, :2] = np.mgrid[:PATTERN_SIZE[0], :PATTERN_SIZE[1]].T.reshape(-1, 2)*SQUARE_SIZE_M
    ok, rv, tv = cv2.solvePnP(obj, corners.reshape(-1, 1, 2), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError('solvePnP failed')
    R, _ = cv2.Rodrigues(rv)
    t = tv.reshape(3)
    if np.any((obj @ R.T + t)[:, 2] <= 0):
        raise ValueError('Board behind camera')
    projected, _ = cv2.projectPoints(obj, rv, tv, K, D)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2, axis=1))))
    # Plane normal in camera coordinates, oriented toward camera center.
    up = R[:, 2].copy()
    if np.dot(up, -t) < 0:
        up *= -1
    h = float(-np.dot(up, t))
    optical = np.array([0., 0., 1.])
    forward = optical - np.dot(optical, up)*up
    if h <= 0 or np.linalg.norm(forward) < 1e-3:
        raise ValueError('Invalid height or nearly vertical optical axis')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    # Ground frame: X right, Y forward, Z up; origin directly under camera.
    R_cg = np.column_stack((right, forward, up))
    t_cg = -h*up
    pitch = float(np.degrees(np.arcsin(np.clip(-np.dot(optical, up), -1, 1))))
    return R_cg, t_cg, h, pitch, rms, rv, tv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=DEVICE)
    parser.add_argument('--image', help='Use an existing RAW ground-board image instead of live capture')
    parser.add_argument('--height', type=float, help='Measured optical-center height in meters (validation only)')
    args = parser.parse_args()
    K, D, Knew = load_intrinsics()
    def save(frame, corners):
        R, t, h, pitch, rms, rv, tv = estimate_pose(corners, K, D)
        H = Knew @ np.column_stack((R[:, 0], R[:, 1], t))
        data = dict(R_cg=R.tolist(), t_cg=t.tolist(), height_m=h, pitch_down_deg=pitch, rms_px=rms,
                    H_ground_to_image=H.tolist(), H_image_to_ground=np.linalg.inv(H).tolist(),
                    intrinsics_sha256=intrinsics_hash(), measured_height_m=args.height,
                    frame='X right, Y forward, Z up; origin below camera; board plane Z=0')
        print(json.dumps(data, indent=2))
        if rms > 0.5:
            print('WARNING: high ground-board RMS; inspect and recapture if needed.')
        if args.height is not None:
            print(f'Measured height difference: {h-args.height:+.3f} m')
            if abs(h-args.height) > max(.03, .05*args.height):
                print('WARNING: height mismatch; check board flatness and pose before using BEV.')
        np.savez(GROUND, R_cg=R, t_cg=t, height_m=h, pitch_down_deg=pitch, rms_px=rms,
                 H_ground_to_image=H, H_image_to_ground=np.linalg.inv(H), intrinsics_sha256=intrinsics_hash())
        (OUTPUT_DIR/'ground_extrinsics.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
        preview = frame.copy()
        cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners.reshape(-1,1,2), True)
        if not cv2.imwrite(str(OUTPUT_DIR/'ground_raw.png'), frame) or not cv2.imwrite(str(OUTPUT_DIR/'ground_preview.png'), preview):
            raise RuntimeError('Failed to save ground images')
        print(f'Saved {GROUND}')
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None or frame.shape[:2] != IMAGE_SIZE[::-1]:
            raise ValueError('Expected a readable raw 1280x720 image')
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if corners is None:
            raise RuntimeError('Checkerboard not found')
        save(frame, corners)
        return
    cap = open_camera(args.camera)
    print('Fix camera in place, lay board flat on floor. S saves pose and exits; Q/Esc cancels.')
    try:
        while True:
            frame = read_frame(cap)
            corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            preview = frame.copy()
            label = 'Board not found'
            if corners is not None:
                cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners.reshape(-1,1,2), True)
                try:
                    _, _, h, pitch, rms, _, _ = estimate_pose(corners, K, D)
                    label = f'h={h:.3f}m pitch={pitch:.1f}deg RMS={rms:.3f}px | S: save'
                except (ValueError, RuntimeError, cv2.error) as e:
                    label = str(e)
            cv2.putText(preview, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            cv2.imshow('Ground calibration', preview)
            key = cv2.waitKey(1)&255
            if key in (ord('q'),27): break
            if key == ord('s') and corners is not None:
                save(frame, corners)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
