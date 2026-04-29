#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <checkpoint.pt> <image_folder_or_video_path> <bundle_dir> [3dgs_args...]"
  echo "Example:"
  echo "  $0 checkpoints/lingbot-map.pt example/loop outputs/gs_bundle --init_max_points 200000 --iterations 7000"
  exit 1
fi

MODEL_PATH="$1"
INPUT_PATH="$2"
BUNDLE_DIR="$3"
shift 3

DEMO_INPUT_ARGS=()
if [[ -d "${INPUT_PATH}" ]]; then
  DEMO_INPUT_ARGS+=(--image_folder "${INPUT_PATH}")
else
  DEMO_INPUT_ARGS+=(--video_path "${INPUT_PATH}")
fi

echo "[1/2] Export lingbot-map -> 3DGS bundle"
python demo.py \
  --model_path "${MODEL_PATH}" \
  "${DEMO_INPUT_ARGS[@]}" \
  --export_3dgs_bundle_dir "${BUNDLE_DIR}" \
  --save_depth_dir "${BUNDLE_DIR}/debug_depth"

echo "[2/2] Run vendored 3DGS training"
python third_party/gaussian_splatting_core/train.py \
  -s "${BUNDLE_DIR}" \
  --dataset_type lingbot_map \
  "$@"
