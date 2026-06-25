#!/usr/bin/env bash
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/Echo-Infinity/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/echo_infinity_juanxi.yaml"
BENCHMARK_CONFIG="config/benchmark.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"
SMOKE_OUTPUT="$WORLD_ROOT/artifacts/predictions/echo-infinity-image-static-smoke"
FULL_OUTPUT="$WORLD_ROOT/artifacts/predictions/echo-infinity-image-static"
LOG_PATH="${1:-$WORLD_ROOT/artifacts/logs/echo_infinity_image_static_$(date +%Y%m%d_%H%M%S).log}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

cd "$WORLD_ROOT"
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[$(date --iso-8601=seconds)] Echo-Infinity image_static GPU 0-7 launcher started"
echo "log_path=$LOG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "smoke_output=$SMOKE_OUTPUT"
echo "full_output=$FULL_OUTPUT"
echo "model_config=$MODEL_CONFIG"

common_args=(
    "$PYTHON_BIN"
    -u
    -m
    worldarena.cli
    generate
    --config
    "$BENCHMARK_CONFIG"
    --model-config
    "$MODEL_CONFIG"
    --manifest
    "$MANIFEST"
    --suites
    image_static
)

echo "[$(date --iso-8601=seconds)] Running Echo-Infinity smoke generation (1 sample)"
"${common_args[@]}" \
    --output-dir "$SMOKE_OUTPUT" \
    --limit 1 \
    --overwrite \
    --fail-fast

echo "[$(date --iso-8601=seconds)] Smoke passed; running full Echo-Infinity image_static generation"
full_args=(
    "${common_args[@]}"
    --output-dir
    "$FULL_OUTPUT"
    --fail-fast
)
if [[ "${WORLDARENA_ECHO_INFINITY_FULL_OVERWRITE:-0}" == "1" ]]; then
    full_args+=(--overwrite)
fi
"${full_args[@]}"

echo "[$(date --iso-8601=seconds)] Echo-Infinity image_static generation completed"
