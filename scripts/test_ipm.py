#!/usr/bin/env python3
"""Static metric BEV from raw camera frames; invalid cells remain unknown."""
import argparse
import cv2
import numpy as np
from calibrate_camera import OUTPUT_DIR, IMAGE_SIZE
from capture_calibration import DEVICE, open_camera, read_frame
from calibrate_ground import GROUND, load_intrinsics, intrinsics_hash


def build_ipm_maps(K, D, R, t, width=4., depth=4., resolution=.05):
    cols, rows = round(width/resolution), round(depth/resolution)
    if min(cols,rows) < 1 or not np.isclose(cols*resolution,width) or not np.isclose(rows*resolution,depth):
        raise ValueError('Map dimensions must be positive multiples of resolution')
    x = (np.arange(cols)+.5)*resolution-width/2
    y = depth-(np.arange(rows)+.5)*resolution
    xx, yy = np.meshgrid(x,y)
    ground = np.stack((xx,yy,np.zeros_like(xx)),axis=-1)
    cam = ground @ R.T + t
    front = cam[...,2] > 1e-6
    # Project directly into distorted RAW image: avoids two resampling passes.
    safe = cam.copy(); safe[~front] = (0.,0.,1.)
    uv,_ = cv2.projectPoints(safe.reshape(-1,3), np.zeros(3), np.zeros(3), K,D)
    uv = uv.reshape(rows,cols,2)
    u,v = uv[...,0],uv[...,1]
    valid = front & np.isfinite(u) & np.isfinite(v) & (u>=0)&(v>=0)&(u<IMAGE_SIZE[0]-1)&(v<IMAGE_SIZE[1]-1)
    # Reject folded/extrapolated radial mapping: check that undistorted ray
    # recovered from projected pixel agrees with the original ground ray.
    idx = np.where(valid)
    if len(idx[0]):
        points = uv[idx].reshape(-1,1,2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        iterative = getattr(cv2, 'undistortPointsIter', None)
        if callable(iterative):
            recovered = iterative(points, K, D, None, None, criteria)
        else:
            # Portable Newton inversion of the five-coefficient pinhole model.
            target = (points.reshape(-1, 2) - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
            recovered = target.copy()
            k1, k2, p1, p2, k3 = np.asarray(D).reshape(-1)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                for _ in range(100):
                    x, y = recovered.T
                    r2 = x*x + y*y
                    radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
                    slope = k1 + 2*k2*r2 + 3*k3*r2*r2
                    fx = x*radial + 2*p1*x*y + p2*(r2+2*x*x) - target[:,0]
                    fy = y*radial + p1*(r2+2*y*y) + 2*p2*x*y - target[:,1]
                    a = radial + 2*x*x*slope + 2*p1*y + 6*p2*x
                    b = 2*x*y*slope + 2*p1*x + 2*p2*y
                    d = radial + 2*y*y*slope + 6*p1*y + 2*p2*x
                    det = a*d-b*b
                    active = (np.abs(det)>1e-12) & np.isfinite(det)
                    dx = np.where(active, (d*fx-b*fy)/det, np.nan)
                    dy = np.where(active, (a*fy-b*fx)/det, np.nan)
                    recovered -= np.column_stack((dx,dy))
                    if np.all(~active | (np.hypot(dx,dy)<1e-10)):
                        break
            check, _ = cv2.projectPoints(np.column_stack((recovered, np.ones(len(recovered)))), np.zeros(3), np.zeros(3), K, D)
            recovered[np.linalg.norm(check.reshape(-1,2)-points.reshape(-1,2),axis=1)>1e-5] = np.nan
        recovered = recovered.reshape(-1,2)
        expected = cam[idx][:,:2]/cam[idx][:,2:3]
        valid[idx] = np.linalg.norm(recovered-expected,axis=1) < .002
    return np.where(valid,u,-1).astype(np.float32), np.where(valid,v,-1).astype(np.float32),valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',type=int,default=DEVICE)
    p.add_argument('--image',help='RAW image; defaults to live camera')
    p.add_argument('--width',type=float,default=4.)
    p.add_argument('--depth',type=float,default=4.)
    p.add_argument('--resolution',type=float,default=.05)
    args=p.parse_args()
    if not all(np.isfinite(v) and v>0 for v in (args.width,args.depth,args.resolution)):
        p.error('Map dimensions and resolution must be positive and finite')
    K,D,_=load_intrinsics()
    with np.load(GROUND,allow_pickle=False) as data:
        if str(data['intrinsics_sha256'].item()) != intrinsics_hash():
            raise RuntimeError('Intrinsics changed; rerun calibrate_ground.py')
        R,t=data['R_cg'],data['t_cg']
    mx,my,valid=build_ipm_maps(K,D,R,t,args.width,args.depth,args.resolution)
    np.savez(OUTPUT_DIR/'ipm_maps.npz',map_x=mx,map_y=my,valid=valid,width_m=args.width,depth_m=args.depth,resolution_m=args.resolution)
    cv2.imwrite(str(OUTPUT_DIR/'ipm_valid_mask.png'),valid.astype(np.uint8)*255)
    print(f'Grid {valid.shape[1]}x{valid.shape[0]}, valid {valid.mean():.1%}. S saves, Q/Esc exits.')
    def render(frame):
        if frame is None or frame.shape[:2]!=IMAGE_SIZE[::-1]: raise ValueError('Expected raw 1280x720 frame')
        bev=cv2.remap(frame,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        bev[~valid]=(100,100,100)
        scale=min(640/bev.shape[1],640/bev.shape[0]); w,h=round(bev.shape[1]*scale),round(bev.shape[0]*scale)
        display=cv2.resize(bev,(w,h),interpolation=cv2.INTER_NEAREST)
        for d in np.arange(.5,args.depth,.5):
            y=round((1-d/args.depth)*h)
            cv2.line(display,(0,y),(w-1,y),(220,220,220),1)
            cv2.putText(display,f'{d:g}m',(4,max(14,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.line(display,(w//2,0),(w//2,h-1),(0,255,255),1)
        return bev,display
    def save(frame,bev,display):
        for name,img in [('ipm_raw.png',frame),('ipm_bev.png',bev),('ipm_preview.png',display)]:
            if not cv2.imwrite(str(OUTPUT_DIR/name),img): raise RuntimeError(f'Cannot save {name}')
    cap=None
    try:
        if args.image:
            frame=cv2.imread(args.image); bev,display=render(frame); save(frame,bev,display)
        else: cap=open_camera(args.camera)
        while True:
            if cap is not None: frame=read_frame(cap); bev,display=render(frame)
            cv2.imshow('Raw RGB',frame); cv2.imshow('Metric BEV (gray = invalid)',display)
            key=cv2.waitKey(30 if cap is None else 1)&255
            if key in (ord('q'),27): break
            if key==ord('s'): save(frame,bev,display); print('Saved IPM preview')
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__': main()
