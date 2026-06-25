#!/usr/bin/env bash
# Launch WorldFM on image_static + image_dynamic with 8 GPUs (decoupled pipeline).
# Model is loaded ONCE per GPU per stage (prepare / final) — no repeated ckpt loading.
#
# Usage:
#   bash scripts/launch_worldfm_juanxi_8gpu.sh
#   bash scripts/launch_worldfm_juanxi_8gpu.sh --limit 100   # smoke test
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
CKPT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin/python"
ENV_BIN_DIR="$(dirname "$PYTHON_BIN")"
MODEL_CONFIG="artifacts/run_configs/worldfm_1001_image_static.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$WORLD_ROOT/artifacts/predictions/worldfm-image-static-dynamic-8gpu-${TIMESTAMP}"
LOG_PATH="$WORLD_ROOT/artifacts/logs/worldfm_8gpu_${TIMESTAMP}.log"

# GPU layout: pipelined full mode runs prepare and final concurrently.
# Split GPUs so both stages can make progress without OOM on the same device.
PREPARE_GPUS="${PREPARE_GPUS:-0,1,2,3}"
FINAL_GPUS="${FINAL_GPUS:-4,5,6,7}"
MERGE_WORKERS="${MERGE_WORKERS:-16}"
THREADS_PER_GPU="${THREADS_PER_GPU:-8}"

# ---------------------------------------------------------------------------
# Local checkpoint paths — avoid HuggingFace Hub downloads
# ---------------------------------------------------------------------------
export WORLDARENA_CHECKPOINT_ROOT="$CKPT_ROOT"
export WORLDFM_PANOGEN_MODEL_PATH="$CKPT_ROOT/FLUX.1-Fill-dev"
export WORLDFM_PANOGEN_LORA_PATH="$CKPT_ROOT/HunyuanWorld-1"
export PATH="$ENV_BIN_DIR:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date --iso-8601=seconds)] WorldFM 8-GPU launcher started"
echo "output_dir=$OUTPUT_DIR"
echo "prepare_gpus=$PREPARE_GPUS"
echo "final_gpus=$FINAL_GPUS"
echo "merge_workers=$MERGE_WORKERS"
echo "log=$LOG_PATH"

cd "$WORLD_ROOT"

# Pass any extra CLI args (e.g. --limit N) through to the decoupled launcher
EXTRA_ARGS=("$@")

"$PYTHON_BIN" scripts/launch_worldfm_decoupled.py \
    --model-config "$MODEL_CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --suites "image_static,image_dynamic" \
    --prepare-gpus "$PREPARE_GPUS" \
    --final-gpus "$FINAL_GPUS" \
    --merge-workers "$MERGE_WORKERS" \
    --threads-per-gpu-worker "$THREADS_PER_GPU" \
    "${EXTRA_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] WorldFM 8-GPU generation completed: $OUTPUT_DIR"
