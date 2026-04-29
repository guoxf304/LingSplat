## Lingbot-Map + 3DGS Integration

This project vendors 3DGS code under `third_party/gaussian_splatting_core` and adds a
custom `LingbotMap` scene reader to avoid large changes in either upstream repository.

### 1) Export bridge bundle

Run `demo.py` with:

```bash
python demo.py \
  --model_path checkpoints/lingbot-map.pt \
  --image_folder example/loop \
  --export_3dgs_bundle_dir outputs/gs_bundle
```

Generated bundle files:

- `images/` (frame images)
- `camera_trajectory_tum.txt` (`frame tx ty tz qx qy qz qw`, c2w)
- `camera_intrinsics.txt` (`frame fx fy cx cy`)
- `init_points.npz` (`xyz`, `rgb`, `frame_id`, optional `conf`)
- `init_points.ply` (for quick visualization/debug)
- `meta.json`

### 2) Train with vendored 3DGS

```bash
python third_party/gaussian_splatting_core/train.py \
  -s outputs/gs_bundle \
  --dataset_type lingbot_map
```

### 3) Random-uniform initialization downsampling

The custom reader supports:

- `--init_sample_ratio` in `(0, 1]` (default `1.0`)
- `--init_max_points` integer cap (default `0`, means no cap)
- `--init_sample_seed` RNG seed (default `0`)

Example:

```bash
python third_party/gaussian_splatting_core/train.py \
  -s outputs/gs_bundle \
  --dataset_type lingbot_map \
  --init_max_points 200000 \
  --init_sample_ratio 0.5 \
  --init_sample_seed 42 \
  --iterations 7000
```

### 4) One-command bridge pipeline

```bash
./scripts/train_3dgs_from_lingbot.sh \
  checkpoints/lingbot-map.pt \
  example/loop \
  outputs/gs_bundle \
  --init_max_points 200000 \
  --iterations 7000
```

### Validation checklist

- Coordinate convention: exported camera is `c2w`; reader converts to `w2c` for camera construction.
- Reproducibility: keep `--init_sample_seed` fixed.
- Memory control: reduce `--init_max_points` and/or lower `--densify_until_iter`.
- If `ModuleNotFoundError: diff_gaussian_rasterization`, install 3DGS CUDA extensions first.
