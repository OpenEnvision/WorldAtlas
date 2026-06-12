#!/usr/bin/env bash
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/lyra/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/lyra1_juanxi.yaml"
BENCHMARK_CONFIG="config/benchmark.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"
SMOKE_OUTPUT="$WORLD_ROOT/artifacts/predictions/lyra-1-image-static-openenvision-gpu4-7-smoke"
FULL_OUTPUT="$WORLD_ROOT/artifacts/predictions/lyra-1-image-static-openenvision-gpu4-7"
LOG_PATH="${1:-$WORLD_ROOT/artifacts/logs/lyra1_image_static_openenvision_gpu4_7_$(date +%Y%m%d_%H%M%S).log}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

cd "$WORLD_ROOT"
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WORLDARENA_LYRA1_POST_RUN_SLEEP_SECONDS="${WORLDARENA_LYRA1_POST_RUN_SLEEP_SECONDS:-5}"

echo "[$(date --iso-8601=seconds)] Lyra1 GPU 4-7 launcher started"
echo "log_path=$LOG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "smoke_output=$SMOKE_OUTPUT"
echo "full_output=$FULL_OUTPUT"
echo "model_config=$MODEL_CONFIG"

while true; do
    gpu_status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")"
    echo "[$(date --iso-8601=seconds)] GPU $CUDA_VISIBLE_DEVICES status:"
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
    echo "[$(date --iso-8601=seconds)] GPU $CUDA_VISIBLE_DEVICES not ready; waiting ${WORLDARENA_LYRA1_WAIT_SECONDS:-300}s"
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
full_args=(
    "${common_args[@]}"
    --output-dir
    "$FULL_OUTPUT"
    --fail-fast
)
if [[ "${WORLDARENA_LYRA1_FULL_OVERWRITE:-0}" == "1" ]]; then
    full_args+=(--overwrite)
fi
"${full_args[@]}"

echo "[$(date --iso-8601=seconds)] Lyra1 GPU 4-7 generation completed"
