"""
Export LingBot-MAP predictions into a 3DGS bridge dataset bundle.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .prediction_saver import save_camera_tum_trajectory


def _write_ply_xyz_rgb(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {xyz.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i]
            r, g, b = rgb_u8[i]
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def _prepare_image_list(
    source_paths: list[str],
    output_images_dir: Path,
    copy_images: bool,
) -> list[str]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for idx, src in enumerate(source_paths):
        src_path = Path(src)
        ext = src_path.suffix.lower() if src_path.suffix else ".png"
        out_name = f"{idx:06d}{ext}"
        out_path = output_images_dir / out_name
        if copy_images:
            shutil.copy2(src_path, out_path)
        else:
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            out_path.symlink_to(src_path.resolve())
        names.append(out_name)
    return names


def export_lingbot_gs_bundle(
    *,
    output_dir: str,
    source_paths: list[str],
    extrinsic_c2w: np.ndarray,
    intrinsic: np.ndarray,
    world_points: np.ndarray,
    images_chw: np.ndarray,
    world_points_conf: Optional[np.ndarray] = None,
    conf_threshold: Optional[float] = None,
    copy_images: bool = True,
) -> None:
    """Export a bundle consumed by custom 3DGS LingBot reader."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images_dir = out / "images"
    frame_names = _prepare_image_list(source_paths, images_dir, copy_images=copy_images)

    save_camera_tum_trajectory(
        extrinsic_c2w=extrinsic_c2w,
        intrinsic=intrinsic,
        output_dir=str(out),
        filename="camera_trajectory_tum.txt",
        intrinsics_filename="camera_intrinsics.txt",
    )

    points = np.asarray(world_points, dtype=np.float32)
    rgb_frames = np.asarray(images_chw, dtype=np.float32)
    rgb_frames = np.transpose(rgb_frames, (0, 2, 3, 1))
    rgb_frames = np.clip(rgb_frames * 255.0, 0.0, 255.0).astype(np.uint8)

    xyz = points.reshape(-1, 3)
    rgb = rgb_frames.reshape(-1, 3)
    frame_ids = np.repeat(np.arange(points.shape[0], dtype=np.int32), points.shape[1] * points.shape[2])

    valid = np.isfinite(xyz).all(axis=1)
    conf_flat = None
    if world_points_conf is not None:
        conf_flat = np.asarray(world_points_conf, dtype=np.float32).reshape(-1)
        if conf_threshold is not None:
            valid = valid & (conf_flat > float(conf_threshold))

    xyz_valid = xyz[valid]
    rgb_valid = rgb[valid]
    frame_ids_valid = frame_ids[valid]
    conf_valid = conf_flat[valid] if conf_flat is not None else None

    np.savez_compressed(
        out / "init_points.npz",
        xyz=xyz_valid.astype(np.float32),
        rgb=rgb_valid.astype(np.uint8),
        frame_id=frame_ids_valid.astype(np.int32),
        conf=(conf_valid.astype(np.float32) if conf_valid is not None else np.array([], dtype=np.float32)),
    )
    _write_ply_xyz_rgb(out / "init_points.ply", xyz_valid, rgb_valid)

    h, w = points.shape[1], points.shape[2]
    meta = {
        "dataset_type": "lingbot_map",
        "version": 1,
        "num_frames": int(points.shape[0]),
        "frame_size_hw": [int(h), int(w)],
        "camera_pose_type": "c2w",
        "camera_traj_file": "camera_trajectory_tum.txt",
        "camera_intrinsics_file": "camera_intrinsics.txt",
        "point_cloud_npz": "init_points.npz",
        "point_cloud_ply": "init_points.ply",
        "images_dir": "images",
        "image_filenames": frame_names,
        "count_rule": (
            f"finite_xyz_and_conf_gt_{conf_threshold}"
            if conf_flat is not None and conf_threshold is not None
            else "finite_xyz"
        ),
    }
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Write a simple quick-check image to ensure image decode path is valid.
    sample_path = images_dir / frame_names[0]
    sample = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    if sample is None:
        raise RuntimeError(f"Failed to read exported image: {sample_path}")
