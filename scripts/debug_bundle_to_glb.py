#!/usr/bin/env python3
"""
Build a debug GLB from a 3DGS bundle:
- Point cloud from init_points.ply
- Camera frustums from camera_trajectory_tum.txt

This helps verify whether points and poses are in the same coordinate frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import trimesh
except ImportError as exc:
    raise SystemExit("trimesh is required: pip install trimesh") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.vis.glb_export import integrate_camera_into_scene


def _load_tum_c2w(path: Path) -> np.ndarray:
    mats = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()
            if not s or s[0] == "frame":
                continue
            tx, ty, tz = map(float, s[1:4])
            qx, qy, qz, qw = map(float, s[4:8])
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rot
            c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
            mats.append(c2w)
    if not mats:
        raise ValueError(f"No valid poses found in {path}")
    return np.stack(mats, axis=0)


def _trajectory_lines(c2w: np.ndarray, radius: float) -> trimesh.Trimesh | None:
    if len(c2w) < 2:
        return None
    pts = c2w[:, :3, 3]
    segs = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-8:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=seg_len, sections=8)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        direction = (p1 - p0) / seg_len
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        if np.linalg.norm(v) < 1e-8:
            rot = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            vx = np.array(
                [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                dtype=np.float64,
            )
            rot = np.eye(3) + vx + vx @ vx / (1.0 + c)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = (p0 + p1) / 2.0
        cyl.apply_transform(T)
        cyl.visual.face_colors[:, :3] = (220, 220, 220)
        segs.append(cyl)
    return trimesh.util.concatenate(segs) if segs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug bundle to GLB")
    parser.add_argument(
        "--bundle_dir",
        type=str,
        default="outputs/depth_frames",
        help="Directory containing init_points.ply and camera_trajectory_tum.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bundle_debug.glb",
        help="Output GLB path",
    )
    parser.add_argument("--cam_scale", type=float, default=1.0, help="Camera frustum scale multiplier")
    parser.add_argument("--traj_radius", type=float, default=0.01, help="Trajectory tube radius")
    parser.add_argument(
        "--pose_convention",
        type=str,
        default="c2w",
        choices=["c2w", "w2c"],
        help="Interpretation of trajectory pose file. Default c2w (no auto conversion).",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    ply_path = bundle_dir / "init_points.ply"
    traj_path = bundle_dir / "camera_trajectory_tum.txt"

    if not ply_path.exists():
        raise FileNotFoundError(f"Missing {ply_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")

    points = None
    colors = None

    # 1) Prefer point-cloud loader
    try:
        pcd = trimesh.load(str(ply_path), force="pointcloud")
        if hasattr(pcd, "vertices"):
            points = np.asarray(pcd.vertices)
        if hasattr(pcd, "colors"):
            colors = np.asarray(pcd.colors)[:, :3]
    except Exception:
        pcd = None

    # 2) Fallback to generic loader (mesh / scene)
    if points is None or points.size == 0:
        pcd = trimesh.load(str(ply_path))
        if isinstance(pcd, trimesh.Trimesh):
            points = np.asarray(pcd.vertices)
            if pcd.visual is not None and hasattr(pcd.visual, "vertex_colors"):
                colors = np.asarray(pcd.visual.vertex_colors)[:, :3]
        elif isinstance(pcd, trimesh.Scene):
            verts = []
            cols = []
            for g in pcd.geometry.values():
                if hasattr(g, "vertices"):
                    v = np.asarray(g.vertices)
                    if v.size > 0:
                        verts.append(v)
                        if hasattr(g, "colors"):
                            c = np.asarray(g.colors)[:, :3]
                            cols.append(c)
                        elif hasattr(g, "visual") and hasattr(g.visual, "vertex_colors"):
                            c = np.asarray(g.visual.vertex_colors)[:, :3]
                            cols.append(c)
            points = np.concatenate(verts, axis=0) if verts else np.zeros((0, 3), dtype=np.float32)
            if cols and sum(len(x) for x in cols) == len(points):
                colors = np.concatenate(cols, axis=0)

    if points is None:
        points = np.zeros((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Unexpected point shape from PLY: {points.shape}")
    if len(points) == 0:
        raise ValueError(
            f"No vertices found in {ply_path}. "
            "Please re-export bundle and ensure init_points.ply is non-empty."
        )

    if colors is None or len(colors) != len(points):
        colors = np.full((len(points), 3), 200, dtype=np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))

    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    scene_scale = max(float(np.linalg.norm(hi - lo)), 0.1)
    cam_size = scene_scale * args.cam_scale

    c2w_raw = _load_tum_c2w(traj_path)
    if args.pose_convention == "w2c":
        c2w = np.linalg.inv(c2w_raw)
        pose_used = "w2c->c2w (manual)"
    else:
        c2w = c2w_raw
        pose_used = "c2w"

    cmap = trimesh.visual.color.interpolate(np.linspace(0, 1, len(c2w)), color_map="viridis")
    for i in range(len(c2w)):
        color = tuple(int(x) for x in cmap[i][:3])
        integrate_camera_into_scene(
            scene=scene,
            transform=c2w[i],
            face_colors=color,
            scene_scale=cam_size,
            frustum_thickness=2.0,
        )

    traj = _trajectory_lines(c2w, radius=args.traj_radius * args.cam_scale)
    if traj is not None:
        scene.add_geometry(traj)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    print(f"Saved debug GLB: {out}")
    print(f"Points: {len(points)}, Cameras: {len(c2w)}")
    print(f"Pose convention used: {pose_used}")


if __name__ == "__main__":
    main()
