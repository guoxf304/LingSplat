"""
Utilities for saving prediction artifacts (depth, point-count stats, camera poses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _normalize_to_u16(depth: np.ndarray) -> np.ndarray:
    """Normalize a depth map to uint16 for visualization-friendly storage."""
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)
    dmin = float(depth.min()) if depth.size > 0 else 0.0
    dmax = float(depth.max()) if depth.size > 0 else 0.0
    if dmax <= dmin:
        return np.zeros_like(depth, dtype=np.uint16)
    scaled = (depth - dmin) / (dmax - dmin)
    return (scaled * 65535.0).astype(np.uint16)


def _rotation_matrix_to_quaternion_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion in [x, y, z, w]."""
    m = rot
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q


def save_depth_frames(
    depth: np.ndarray,
    output_dir: str,
    source_paths: Optional[list[str]] = None,
    save_npy: bool = True,
    save_png_u16: bool = True,
) -> None:
    """Save per-frame depth maps to disk."""
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape [S,H,W] or [S,H,W,1], got {depth.shape}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npy_dir = out / "npy"
    png_dir = out / "png"
    if save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)
    if save_png_u16:
        png_dir.mkdir(parents=True, exist_ok=True)

    if source_paths is not None and len(source_paths) != depth.shape[0]:
        raise ValueError(
            f"source_paths length ({len(source_paths)}) must match depth frames ({depth.shape[0]})"
        )

    for i in range(depth.shape[0]):
        stem = f"{i:06d}" if source_paths is None else Path(source_paths[i]).stem
        frame_depth = depth[i].astype(np.float32, copy=False)
        if save_npy:
            np.save(str(npy_dir / f"{stem}.npy"), frame_depth)
        if save_png_u16:
            depth_u16 = _normalize_to_u16(frame_depth)
            cv2.imwrite(str(png_dir / f"{stem}.png"), depth_u16)


def save_point_cloud_counts(
    world_points: np.ndarray,
    output_dir: str,
    source_paths: Optional[list[str]] = None,
    world_points_conf: Optional[np.ndarray] = None,
    conf_threshold: Optional[float] = None,
    filename: str = "point_cloud_counts.txt",
) -> None:
    """Save per-frame point-cloud counts and total count to a txt file."""
    world_points = np.asarray(world_points)
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError(f"Expected world_points shape [S,H,W,3], got {world_points.shape}")

    if source_paths is not None and len(source_paths) != world_points.shape[0]:
        raise ValueError(
            f"source_paths length ({len(source_paths)}) must match point frames ({world_points.shape[0]})"
        )

    conf = None
    if world_points_conf is not None:
        conf = np.asarray(world_points_conf)
        if conf.shape != world_points.shape[:3]:
            raise ValueError(
                f"world_points_conf shape {conf.shape} must match world_points[:3] {world_points.shape[:3]}"
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / filename

    per_frame_counts = []
    for i in range(world_points.shape[0]):
        pts = world_points[i]
        valid = np.isfinite(pts).all(axis=-1)
        if conf is not None and conf_threshold is not None:
            valid = valid & (conf[i] > conf_threshold)
        per_frame_counts.append(int(valid.sum()))

    total_count = int(sum(per_frame_counts))
    total_frames = int(world_points.shape[0])

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("# Point cloud count statistics\n")
        f.write(f"total_frames: {total_frames}\n")
        if conf is not None and conf_threshold is not None:
            f.write(f"count_rule: finite_xyz_and_conf_gt_{conf_threshold}\n")
        else:
            f.write("count_rule: finite_xyz\n")
        f.write(f"total_points_all_frames: {total_count}\n\n")
        f.write("per_frame_counts:\n")
        for i, count in enumerate(per_frame_counts):
            frame_name = f"{i:06d}" if source_paths is None else Path(source_paths[i]).stem
            f.write(f"{i:06d}\t{frame_name}\t{count}\n")


def save_camera_tum_trajectory(
    extrinsic_c2w: np.ndarray,
    intrinsic: np.ndarray,
    output_dir: str,
    filename: str = "camera_trajectory_tum.txt",
    intrinsics_filename: str = "camera_intrinsics.txt",
) -> None:
    """
    Save camera trajectory in TUM-like format with header.

    Output columns:
        frame tx ty tz qx qy qz qw
    The first line is a header. Intrinsics are saved separately.
    """
    extrinsic_c2w = np.asarray(extrinsic_c2w)
    intrinsic = np.asarray(intrinsic)
    if extrinsic_c2w.ndim != 3 or extrinsic_c2w.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape [S,3,4], got {extrinsic_c2w.shape}")
    if intrinsic.ndim != 3 or intrinsic.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsic shape [S,3,3], got {intrinsic.shape}")
    if extrinsic_c2w.shape[0] != intrinsic.shape[0]:
        raise ValueError("extrinsic and intrinsic frame counts must match")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / filename
    intrinsics_path = out / intrinsics_filename

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("frame tx ty tz qx qy qz qw\n")

        for i in range(extrinsic_c2w.shape[0]):
            rt = extrinsic_c2w[i]
            rot = rt[:, :3].astype(np.float64, copy=False)
            trans = rt[:, 3].astype(np.float64, copy=False)
            qx, qy, qz, qw = _rotation_matrix_to_quaternion_xyzw(rot)
            f.write(
                f"{i:06d} "
                f"{trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f} "
                f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
            )

    with intrinsics_path.open("w", encoding="utf-8") as f:
        f.write("frame fx fy cx cy\n")
        for i in range(intrinsic.shape[0]):
            k = intrinsic[i]
            f.write(f"{i:06d} {k[0,0]:.8f} {k[1,1]:.8f} {k[0,2]:.8f} {k[1,2]:.8f}\n")
