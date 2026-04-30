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
from scipy.spatial.transform import Rotation

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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i]
            r, g, b = rgb_u8[i]
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")
    tmp_path.replace(path)


def _random_uniform_downsample(
    *,
    xyz: np.ndarray,
    rgb: np.ndarray,
    conf: Optional[np.ndarray],
    sample_ratio: float,
    max_points: int,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if sample_ratio <= 0.0 or sample_ratio > 1.0:
        raise ValueError(f"sample_ratio must be in (0,1], got {sample_ratio}")
    if max_points < 0:
        raise ValueError(f"max_points must be >= 0, got {max_points}")

    n = xyz.shape[0]
    target = n
    if sample_ratio < 1.0:
        target = max(1, int(np.floor(n * sample_ratio)))
    if max_points > 0:
        target = min(target, max_points)
    if target >= n:
        return xyz, rgb, conf, np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(sample_seed)
    idx = rng.choice(n, size=target, replace=False)
    xyz_ds = xyz[idx]
    rgb_ds = rgb[idx]
    conf_ds = conf[idx] if conf is not None else None
    return xyz_ds, rgb_ds, conf_ds, idx


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
    init_sample_ratio: float = 0.03125,
    init_max_points: int = 0,
    init_sample_seed: int = 0,
    point_space: str = "auto",
    apply_viewer_alignment: bool = True,
    use_viewer_point_logic: bool = False,
    viewer_downsample_factor: int = 32,
    debug_first_n_frames: int = 0,
    trajectory_use_viewer_logic: bool = True,
) -> None:
    """Export a bundle consumed by custom 3DGS LingBot reader."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    points = np.asarray(world_points, dtype=np.float32)
    extrinsic_c2w = np.asarray(extrinsic_c2w, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    rgb_frames = np.asarray(images_chw, dtype=np.float32)

    if debug_first_n_frames and debug_first_n_frames > 0:
        n = int(min(debug_first_n_frames, points.shape[0]))
        source_paths = source_paths[:n]
        points = points[:n]
        extrinsic_c2w = extrinsic_c2w[:n]
        intrinsic = intrinsic[:n]
        rgb_frames = rgb_frames[:n]
        if world_points_conf is not None:
            world_points_conf = np.asarray(world_points_conf)[:n]

    images_dir = out / "images"
    frame_names = _prepare_image_list(source_paths, images_dir, copy_images=copy_images)
    rgb_frames = np.transpose(rgb_frames, (0, 2, 3, 1))
    rgb_frames = np.clip(rgb_frames * 255.0, 0.0, 255.0).astype(np.uint8)

    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected world_points shape [S,H,W,3], got {points.shape}")
    if extrinsic_c2w.ndim != 3 or extrinsic_c2w.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic_c2w shape [S,3,4], got {extrinsic_c2w.shape}")
    if extrinsic_c2w.shape[0] != points.shape[0]:
        raise ValueError("Frame count mismatch between world_points and extrinsic_c2w")

    applied_camera_to_world = False
    point_space_used = point_space
    if point_space not in {"auto", "camera", "world"}:
        raise ValueError(f"point_space must be one of auto/camera/world, got {point_space}")

    # Build trajectory poses with viewer-compatible rule, while keeping point
    # processing on the original extrinsic path.
    if trajectory_use_viewer_logic:
        extr_4x4 = np.zeros((extrinsic_c2w.shape[0], 4, 4), dtype=np.float32)
        extr_4x4[:, :3, :4] = extrinsic_c2w
        extr_4x4[:, 3, 3] = 1.0
        traj_c2w = np.linalg.inv(extr_4x4)[:, :3, :4]
    else:
        traj_c2w = extrinsic_c2w.copy()

    if point_space == "auto":
        pts_flat = points.reshape(-1, 3)
        pts_valid = pts_flat[np.isfinite(pts_flat).all(axis=1)]
        point_extent = 0.0
        if pts_valid.size > 0:
            pmin = pts_valid.min(axis=0)
            pmax = pts_valid.max(axis=0)
            point_extent = float(np.linalg.norm(pmax - pmin))
        traj_t = extrinsic_c2w[:, :, 3]
        traj_extent = float(np.linalg.norm(traj_t.max(axis=0) - traj_t.min(axis=0)))
        # Heuristic: points collapse near origin while camera trajectory spans meters.
        if point_extent < 0.25 and traj_extent > 1.0:
            point_space_used = "camera"
        else:
            point_space_used = "world"

    if point_space_used == "camera":
        # Convert per-frame camera-space points to world space: p_w = R * p_c + t.
        r = extrinsic_c2w[:, :3, :3]
        t = extrinsic_c2w[:, :3, 3]
        points = np.einsum("sij,shwj->shwi", r, points) + t[:, None, None, :]
        applied_camera_to_world = True

    if apply_viewer_alignment:
        # Match GLB/PLY exporter scene transform:
        # T = inv(w2c_0) @ opengl_flip_yz @ rot_y_180
        # and apply to both points and camera c2w.
        opengl = np.eye(4, dtype=np.float32)
        opengl[1, 1] = -1.0
        opengl[2, 2] = -1.0
        rot_y_180 = np.eye(4, dtype=np.float32)
        rot_y_180[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix().astype(np.float32)

        c2w0 = np.eye(4, dtype=np.float32)
        c2w0[:3, :4] = extrinsic_c2w[0]
        align_t = c2w0 @ opengl @ rot_y_180

        # Transform points: p' = T * p.
        points_h = np.concatenate(
            [points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)],
            axis=-1,
        )
        points = np.einsum("ij,shwj->shwi", align_t, points_h)[..., :3]

        # Transform camera poses: c2w' = T * c2w.
        c2w_4x4 = np.zeros((extrinsic_c2w.shape[0], 4, 4), dtype=np.float32)
        c2w_4x4[:, :3, :4] = extrinsic_c2w
        c2w_4x4[:, 3, 3] = 1.0
        c2w_4x4 = np.einsum("ij,sjk->sik", align_t, c2w_4x4)
        extrinsic_c2w = c2w_4x4[:, :3, :4]

        traj_4x4 = np.zeros((traj_c2w.shape[0], 4, 4), dtype=np.float32)
        traj_4x4[:, :3, :4] = traj_c2w
        traj_4x4[:, 3, 3] = 1.0
        traj_4x4 = np.einsum("ij,sjk->sik", align_t, traj_4x4)
        traj_c2w = traj_4x4[:, :3, :4]

    # IMPORTANT: write trajectory/intrinsics after all coordinate transforms,
    # so camera files and exported points stay in the same frame.
    save_camera_tum_trajectory(
        extrinsic_c2w=traj_c2w,
        intrinsic=intrinsic,
        output_dir=str(out),
        filename="camera_trajectory_tum.txt",
        intrinsics_filename="camera_intrinsics.txt",
    )

    if use_viewer_point_logic:
        # Match PointCloudViewer.parse_pc_data behavior:
        # finite filter + conf threshold, then per-frame stride downsample.
        ds = int(max(1, viewer_downsample_factor))
        xyz_valid_list = []
        rgb_valid_list = []
        fid_valid_list = []
        conf_valid_list = []

        xyz_export_list = []
        rgb_export_list = []
        fid_export_list = []
        conf_export_list = []

        for s in range(points.shape[0]):
            xyz_s = points[s].reshape(-1, 3)
            rgb_s = rgb_frames[s].reshape(-1, 3)
            valid_s = np.isfinite(xyz_s).all(axis=1)

            conf_s = None
            if world_points_conf is not None:
                conf_s = np.asarray(world_points_conf[s], dtype=np.float32).reshape(-1)
                if conf_threshold is not None:
                    valid_s = valid_s & (conf_s > float(conf_threshold))

            xyz_s_valid = xyz_s[valid_s]
            rgb_s_valid = rgb_s[valid_s]
            fid_s_valid = np.full(xyz_s_valid.shape[0], s, dtype=np.int32)
            conf_s_valid = conf_s[valid_s] if conf_s is not None else None

            xyz_valid_list.append(xyz_s_valid)
            rgb_valid_list.append(rgb_s_valid)
            fid_valid_list.append(fid_s_valid)
            if conf_s_valid is not None:
                conf_valid_list.append(conf_s_valid)

            idx_s = np.arange(0, xyz_s_valid.shape[0], ds, dtype=np.int64)
            xyz_export_list.append(xyz_s_valid[idx_s])
            rgb_export_list.append(rgb_s_valid[idx_s])
            fid_export_list.append(fid_s_valid[idx_s])
            if conf_s_valid is not None:
                conf_export_list.append(conf_s_valid[idx_s])

        xyz_valid = np.concatenate(xyz_valid_list, axis=0) if xyz_valid_list else np.zeros((0, 3), dtype=np.float32)
        rgb_valid = np.concatenate(rgb_valid_list, axis=0) if rgb_valid_list else np.zeros((0, 3), dtype=np.uint8)
        frame_ids_valid = np.concatenate(fid_valid_list, axis=0) if fid_valid_list else np.zeros((0,), dtype=np.int32)
        conf_valid = np.concatenate(conf_valid_list, axis=0) if conf_valid_list else None

        xyz_export = np.concatenate(xyz_export_list, axis=0) if xyz_export_list else np.zeros((0, 3), dtype=np.float32)
        rgb_export = np.concatenate(rgb_export_list, axis=0) if rgb_export_list else np.zeros((0, 3), dtype=np.uint8)
    else:
        xyz = points.reshape(-1, 3)
        rgb = rgb_frames.reshape(-1, 3)
        frame_ids = np.repeat(np.arange(points.shape[0], dtype=np.int32), points.shape[1] * points.shape[2])

        conf_flat = None
        if world_points_conf is not None:
            conf_flat = np.asarray(world_points_conf, dtype=np.float32).reshape(-1)

        valid = np.isfinite(xyz).all(axis=1)
        if conf_flat is not None and conf_threshold is not None:
            valid = valid & (conf_flat > float(conf_threshold))

        xyz_valid = xyz[valid]
        rgb_valid = rgb[valid]
        frame_ids_valid = frame_ids[valid]
        conf_valid = conf_flat[valid] if conf_flat is not None else None

        xyz_export, rgb_export, _, _ = _random_uniform_downsample(
            xyz=xyz_valid,
            rgb=rgb_valid,
            conf=conf_valid,
            sample_ratio=float(init_sample_ratio),
            max_points=int(init_max_points),
            sample_seed=int(init_sample_seed),
        )

    _write_ply_xyz_rgb(out / "init_points.ply", xyz_export, rgb_export)

    h, w = points.shape[1], points.shape[2]
    meta = {
        "dataset_type": "lingbot_map",
        "version": 1,
        "num_frames": int(points.shape[0]),
        "frame_size_hw": [int(h), int(w)],
        "camera_pose_type": "c2w",
        "camera_traj_file": "camera_trajectory_tum.txt",
        "camera_intrinsics_file": "camera_intrinsics.txt",
        "point_cloud_ply": "init_points.ply",
        "images_dir": "images",
        "image_filenames": frame_names,
        "count_rule": (
            f"finite_xyz_and_conf_gt_{conf_threshold}"
            if world_points_conf is not None and conf_threshold is not None
            else "finite_xyz"
        ),
        "points_before_downsample": int(xyz_valid.shape[0]),
        "points_after_downsample": int(xyz_export.shape[0]),
        "init_sample_ratio": float(init_sample_ratio),
        "init_max_points": int(init_max_points),
        "init_sample_seed": int(init_sample_seed),
        "point_space_arg": point_space,
        "point_space_used": point_space_used,
        "applied_camera_to_world": bool(applied_camera_to_world),
        "apply_viewer_alignment": bool(apply_viewer_alignment),
        "use_viewer_point_logic": bool(use_viewer_point_logic),
        "viewer_downsample_factor": int(max(1, viewer_downsample_factor)),
        "trajectory_use_viewer_logic": bool(trajectory_use_viewer_logic),
    }
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Write a simple quick-check image to ensure image decode path is valid.
    sample_path = images_dir / frame_names[0]
    sample = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    if sample is None:
        raise RuntimeError(f"Failed to read exported image: {sample_path}")
