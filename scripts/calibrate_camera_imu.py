#!/usr/bin/env python3
"""Fit camera-to-IMU rotation and the tip-to-camera vector from saved poses."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from camera_imu_geometry import (
    EXTRINSICS_PATH,
    INTRINSICS_PATH,
    MIN_DIVERSITY,
    MIN_PAIRS,
    PAIR_DIR,
    TIP_DISTANCE_M,
    load_intrinsics,
    load_pairs,
    pose_diversity,
    quaternion_rotation,
)


def fit_rotation(up_camera, up_imu):
    """Find the proper rotation R_ic such that R_ic @ up_camera = up_imu."""
    u, _, vt = np.linalg.svd(up_imu.T @ up_camera)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    return u @ correction @ vt


def fit_tip_vector(up_imu, heights):
    vector, _, rank, _ = np.linalg.lstsq(up_imu, heights, rcond=None)
    if rank != 3:
        raise ValueError("The height fit needs more front and side tilt variation")
    return vector


def rotation_errors(rotation, up_camera, up_imu):
    cosine = np.sum((up_camera @ rotation.T) * up_imu, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def fit_extrinsics(pairs):
    if len(pairs) < MIN_PAIRS:
        raise ValueError(
            f"Capture at least {MIN_PAIRS} distinct poses. Found {len(pairs)}."
        )
    records = [record for _, record in pairs]
    up_camera = np.asarray([record["up_camera"] for record in records], dtype=float)
    heights = np.asarray([record["height_m"] for record in records], dtype=float)
    if up_camera.shape != (len(records), 3) or not np.allclose(
        np.linalg.norm(up_camera, axis=1), 1.0, atol=1e-3
    ):
        raise ValueError("Expected a finite unit camera-plane normal for each pose")
    if heights.shape != (len(records),) or not np.all(
        np.isfinite(heights) & (heights > 0)
    ):
        raise ValueError("Expected a positive camera height for each pose")
    orientations = np.stack([quaternion_rotation(r["q_wxyz"]) for r in records])
    up_imu = orientations[:, 2, :]
    diversity = min(pose_diversity(up_camera), pose_diversity(up_imu))
    if diversity < MIN_DIVERSITY:
        raise ValueError(
            f"Pose diversity {diversity:.3f} < {MIN_DIVERSITY:.2f}. "
            "Capture more front and side tilt."
        )

    rotation = fit_rotation(up_camera, up_imu)
    tip_vector = fit_tip_vector(up_imu, heights)
    angle_errors = rotation_errors(rotation, up_camera, up_imu)
    height_errors = up_imu @ tip_vector - heights
    inverse_up = orientations[:, :, 2]
    inverse_rms = rms(
        rotation_errors(fit_rotation(up_camera, inverse_up), up_camera, inverse_up)
    )

    held_angles, held_heights = [], []
    for index in range(len(pairs)):
        keep = np.arange(len(pairs)) != index
        held_rotation = fit_rotation(up_camera[keep], up_imu[keep])
        held_vector = fit_tip_vector(up_imu[keep], heights[keep])
        held_angles.append(
            rotation_errors(
                held_rotation, up_camera[index : index + 1], up_imu[index : index + 1]
            )[0]
        )
        held_heights.append(up_imu[index] @ held_vector - heights[index])

    return {
        "R_ic": rotation,
        "tip_to_camera_imu_m": tip_vector,
        "pose_diversity": diversity,
        "rotation_rms_deg": rms(angle_errors),
        "height_rms_m": rms(height_errors),
        "inverse_convention_rms_deg": inverse_rms,
        "loo_rotation_deg": np.asarray(held_angles),
        "loo_height_m": np.asarray(held_heights),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=PAIR_DIR)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--output", type=Path, default=EXTRINSICS_PATH)
    args = parser.parse_args()
    try:
        intrinsics = load_intrinsics(args.intrinsics)
        pairs = load_pairs(args.pairs, intrinsics.digest)
        fit = fit_extrinsics(pairs)
        angle_rms, height_rms = fit["rotation_rms_deg"], fit["height_rms_m"]
        angles, heights = fit["loo_rotation_deg"], fit["loo_height_m"]
        print(f"{len(pairs)} poses. Diversity {fit['pose_diversity']:.3f}.")
        print(f"Fit RMS {angle_rms:.2f} deg / {height_rms * 100:.1f} cm.")
        print(f"Leave-one-out RMS {rms(angles):.2f} deg / {rms(heights) * 100:.1f} cm.")
        for (name, _), angle, height in zip(pairs, angles, heights, strict=True):
            print(f"  {name}  {angle:.2f} deg  {height * 100:+.1f} cm")
        length = float(np.linalg.norm(fit["tip_to_camera_imu_m"]))
        print(
            f"Fitted tip-to-camera length {length * 100:.1f} cm. "
            f"Measured reference {TIP_DISTANCE_M * 100:.1f} cm."
        )
        if abs(length - TIP_DISTANCE_M) > 0.04:
            print("Check that the tip stayed on the floor and verify the square size.")
        if fit["inverse_convention_rms_deg"] + 0.3 < angle_rms:
            raise ValueError(
                "IMU orientation convention is inconsistent with the checkerboard"
            )
        if (
            angle_rms > 2.0
            or angles.max() > 4.0
            or rms(heights) > 0.025
            or np.max(np.abs(heights)) > 0.05
        ):
            raise ValueError(
                "Pose validation failed. Recapture the poses with the largest errors."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=args.output.parent, suffix=".npz", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            try:
                np.savez(
                    temp,
                    **fit,
                    intrinsics_sha256=intrinsics.digest,
                    pair_names=[name for name, _ in pairs],
                    nominal_tip_distance_m=TIP_DISTANCE_M,
                )
                temp.close()
                temp_path.replace(args.output)
            finally:
                temp_path.unlink(missing_ok=True)
        print(f"Wrote {args.output}")
    except (KeyError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
