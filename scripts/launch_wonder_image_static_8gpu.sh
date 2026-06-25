#!/usr/bin/env bash
# Launch WorldArena image_static generation for WonderWorld and WonderJourney.
#
# Default behavior:
#   1. Run WonderWorld on 8 shards, one GPU per shard.
#   2. After all WonderWorld shards finish, run WonderJourney the same way.
#
# Each shard owns manifest[shard_index::num_shards]. The WorldArena adapters use
# a batch runner so each shard keeps checkpoints/models resident while processing
# its whole shard instead of reloading for every sample.
#
# Usage:
#   bash scripts/launch_wonder_image_static_8gpu.sh
#   RUN_ID=myrun LIMIT_PER_SHARD=1 bash scripts/launch_wonder_image_static_8gpu.sh
#   MODEL_LIST=wonderjourney bash scripts/launch_wonder_image_static_8gpu.sh
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/ac3d/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
BENCHMARK_CONFIG="config/benchmark_openenvision.yaml"
MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"

NUM_SHARDS="${NUM_SHARDS:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_LIST="${MODEL_LIST:-wonderworld wonderjourney}"
LIMIT_PER_SHARD="${LIMIT_PER_SHARD:-}"
OUTPUT_BASE="${OUTPUT_BASE:-$WORLD_ROOT/artifacts/predictions}"
LOG_BASE="${LOG_BASE:-$WORLD_ROOT/artifacts/logs}"
OVERWRITE="${OVERWRITE:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"

status_file="$LOG_BASE/wonder_image_static_8gpu_${RUN_ID}.status"
: > "$status_file"

log_status() {
    local msg="$1"
    echo "[$(date --iso-8601=seconds)] $msg" | tee -a "$status_file"
}

model_config_for() {
    case "$1" in
        wonderworld) echo "config/models/wonderworld.yaml" ;;
        wonderjourney) echo "config/models/wonderjourney.yaml" ;;
        *) echo "unknown model: $1" >&2; return 2 ;;
    esac
}

run_model() {
    local model="$1"
    local model_config
    model_config="$(model_config_for "$model")"
    local output_dir="$OUTPUT_BASE/${model}-image-static-8gpu-singleload-${RUN_ID}"
    local log_dir="$LOG_BASE/${model}_image_static_8gpu_${RUN_ID}"
    mkdir -p "$output_dir" "$log_dir"

    log_status "starting model=$model output_dir=$output_dir log_dir=$log_dir num_shards=$NUM_SHARDS limit=${LIMIT_PER_SHARD:-<none>}"
    {
        echo "RUN_ID=$RUN_ID"
        echo "MODEL=$model"
        echo "OUTPUT_DIR=$output_dir"
        echo "LOG_DIR=$log_dir"
        echo "NUM_SHARDS=$NUM_SHARDS"
        echo "LIMIT_PER_SHARD=${LIMIT_PER_SHARD:-}"
        echo "MODEL_CONFIG=$model_config"
        echo "MANIFEST=$MANIFEST"
    } > "$log_dir/run_meta.env"

    local pids=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        local shard_log="$log_dir/shard${i}.log"
        local args=(
            "$PYTHON_BIN" -u -m worldarena.cli generate
            --config "$BENCHMARK_CONFIG"
            --model-config "$model_config"
            --manifest "$MANIFEST"
            --suites image_static
            --output-dir "$output_dir"
            --num-shards "$NUM_SHARDS"
            --shard-index "$i"
        )
        if [[ -n "$LIMIT_PER_SHARD" ]]; then
            args+=(--limit "$LIMIT_PER_SHARD")
        fi
        if [[ "$OVERWRITE" == "1" ]]; then
            args+=(--overwrite)
        fi
        if [[ "$FAIL_FAST" == "1" ]]; then
            args+=(--fail-fast)
        fi
        log_status "launching model=$model shard=$i gpu=$i log=$shard_log"
        env CUDA_VISIBLE_DEVICES="$i" nohup "${args[@]}" > "$shard_log" 2>&1 &
        pids+=("$!")
    done

    printf '%s\n' "${pids[@]}" > "$log_dir/shard_pids.txt"
    log_status "launched model=$model pids=${pids[*]}"

    local fail=0
    for idx in "${!pids[@]}"; do
        if wait "${pids[$idx]}"; then
            log_status "completed model=$model shard=$idx pid=${pids[$idx]} ok"
        else
            local rc=$?
            fail=1
            log_status "failed model=$model shard=$idx pid=${pids[$idx]} rc=$rc"
        fi
    done

    local generated_count
    generated_count="$(find "$output_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
    log_status "finished model=$model generated_mp4=$generated_count output_dir=$output_dir"
    return "$fail"
}

log_status "launcher started RUN_ID=$RUN_ID MODEL_LIST=$MODEL_LIST"
for model in $MODEL_LIST; do
    run_model "$model"
done
log_status "launcher finished RUN_ID=$RUN_ID"
