#!/usr/bin/env bash
# Attach final-stage workers to an in-flight decoupled WorldFM run.
# Use when an older launcher is still waiting for prepare to finish but merge
# artifacts are already available and you want mp4 outputs to appear early.
#
# Example:
#   bash scripts/attach_worldfm_final.sh \
#     /mnt/cpfs/.../worldfm-image-static-dynamic-8gpu-20260612_135055 \
#     /mnt/cpfs/.../worldfm-image-static-dynamic-8gpu-20260612_135055/_worldfm_decoupled/run_157229129b
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <output_dir> <work_root> [extra launch_worldfm_decoupled.py args...]" >&2
  exit 1
fi

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin/python"
OUTPUT_DIR="$1"
WORK_ROOT="$2"
shift 2

SPEC_PATH="${WORK_ROOT}/requests.jsonl"
if [[ ! -f "$SPEC_PATH" ]]; then
  echo "missing batch spec: $SPEC_PATH" >&2
  exit 1
fi

FINAL_GPUS="${FINAL_GPUS:-4,5,6,7}"
THREADS_PER_GPU="${THREADS_PER_GPU:-8}"
MODEL_CONFIG="${MODEL_CONFIG:-artifacts/run_configs/worldfm_1001_image_static.yaml}"

cd "$WORLD_ROOT"
exec "$PYTHON_BIN" scripts/launch_worldfm_decoupled.py \
  --model-config "$MODEL_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --mode final \
  --work-root "$WORK_ROOT" \
  --spec-path "$SPEC_PATH" \
  --final-gpus "$FINAL_GPUS" \
  --threads-per-gpu-worker "$THREADS_PER_GPU" \
  "$@"
