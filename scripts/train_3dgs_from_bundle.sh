#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <bundle_dir> [3dgs_args...]"
  echo "Example:"
  echo "  $0 outputs/depth_frames --init_max_points 200000 --iterations 7000"
  exit 1
fi

BUNDLE_DIR="$1"
shift

echo "Run 3DGS training from bundle: ${BUNDLE_DIR}"
python third_party/gaussian_splatting_core/train.py \
  -s "${BUNDLE_DIR}" \
  --dataset_type lingbot_map \
  "$@"
