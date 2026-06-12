#!/usr/bin/env bash
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/lyra/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/lyra1_juanxi.yaml"
BENCHMARK_CONFIG="config/benchmark.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"
SMOKE_OUTPUT="$WORLD_ROOT/artifacts/predictions/lyra-1-image-static-openenvision-v2-smoke"
FULL_OUTPUT="$WORLD_ROOT/artifacts/predictions/lyra-1-image-static-openenvision-v2"
LOG_PATH="${1:-$WORLD_ROOT/artifacts/logs/lyra1_image_static_openenvision_v2_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

cd "$WORLD_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export WORLDARENA_LYRA1_POST_RUN_SLEEP_SECONDS="${WORLDARENA_LYRA1_POST_RUN_SLEEP_SECONDS:-5}"

echo "[$(date --iso-8601=seconds)] Lyra1 v2 launcher started"
echo "log_path=$LOG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "smoke_output=$SMOKE_OUTPUT"
echo "full_output=$FULL_OUTPUT"

while true; do
    gpu_status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3)"
    echo "[$(date --iso-8601=seconds)] GPU 0-3 status:"
    echo "$gpu_status"
    if awk -F, '
        {
            gsub(/ /, "", $2);
            gsub(/ /, "", $3);
            if (($2 + 0) > 5000 || ($3 + 0) > 20) {
                bad = 1;
            }
        }
        END { exit bad }
    ' <<< "$gpu_status"; then
        break
    fi
    echo "[$(date --iso-8601=seconds)] GPU 0-3 not ready; waiting ${WORLDARENA_LYRA1_WAIT_SECONDS:-300}s"
    sleep "${WORLDARENA_LYRA1_WAIT_SECONDS:-300}"
done

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

echo "[$(date --iso-8601=seconds)] Running Lyra1 smoke generation"
"${common_args[@]}" \
    --output-dir "$SMOKE_OUTPUT" \
    --limit 1 \
    --overwrite \
    --fail-fast

echo "[$(date --iso-8601=seconds)] Smoke passed; running full Lyra1 image_static generation"
"${common_args[@]}" \
    --output-dir "$FULL_OUTPUT" \
    --overwrite \
    --fail-fast

echo "[$(date --iso-8601=seconds)] Lyra1 v2 generation completed"
