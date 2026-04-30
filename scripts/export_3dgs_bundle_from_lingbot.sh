#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <checkpoint.pt> <image_folder_or_video_path> <bundle_dir> [demo_args...]"
  echo "Example:"
  echo "  $0 checkpoints/lingbot-map.pt example/loop outputs/depth_frames --use_sdpa"
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

echo "Export Lingbot-Map predictions to 3DGS bundle: ${BUNDLE_DIR}"
python demo.py \
  --model_path "${MODEL_PATH}" \
  "${DEMO_INPUT_ARGS[@]}" \
  --save_depth_dir "${BUNDLE_DIR}" \
  --export_3dgs_bundle_dir "${BUNDLE_DIR}" \
  --downsample_factor 32 \
  --export_3dgs_point_space auto \
  --export_3dgs_points_source depth \
  --export_3dgs_use_viewer_point_logic \
  "$@"

echo "Done. You can now restart and run 3DGS training from ${BUNDLE_DIR}."
