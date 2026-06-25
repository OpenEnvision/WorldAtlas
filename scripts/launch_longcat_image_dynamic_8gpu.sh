#!/usr/bin/env bash
# Launch LongCat-Video (I2V) on the WorldArena image_dynamic suite using 8 GPUs.
#
# Strategy: 8 independent single-GPU shards. Shard i runs on GPU i and processes
# manifest[i::8] of the image_dynamic samples. The worldarena CLI orchestrates;
# the LongCat batch runner (config python_bin) is spawned per shard and loads the
# checkpoint ONCE, then generates every sample in the shard sequentially.
#
# Usage:
#   bash scripts/launch_longcat_image_dynamic_8gpu.sh
#   LONGCAT_LIMIT=2 bash scripts/launch_longcat_image_dynamic_8gpu.sh   # smoke (per shard limit)
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/longcat_video.yaml"
BENCHMARK_CONFIG="config/benchmark_openenvision.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"

NUM_SHARDS="${NUM_SHARDS:-8}"
LONGCAT_LIMIT="${LONGCAT_LIMIT:-}"   # optional --limit passed to each shard (smoke testing)

TIMESTAMP="${LONGCAT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${LONGCAT_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/longcat-video-image-dynamic-8gpu-${TIMESTAMP}}"
LOG_DIR="$WORLD_ROOT/artifacts/logs/longcat_image_dynamic_8gpu_${TIMESTAMP}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[$(date --iso-8601=seconds)] LongCat-Video image_dynamic 8-GPU launcher started"
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_DIR"
echo "num_shards=$NUM_SHARDS"
echo "limit_per_shard=${LONGCAT_LIMIT:-<none>}"
echo "orchestration_python=$PYTHON_BIN"

pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
    shard_log="$LOG_DIR/shard${i}.log"
    args=(
        "$PYTHON_BIN" -u -m worldarena.cli generate
        --config "$BENCHMARK_CONFIG"
        --model-config "$MODEL_CONFIG"
        --manifest "$MANIFEST"
        --suites image_dynamic
        --output-dir "$OUTPUT_DIR"
        --num-shards "$NUM_SHARDS"
        --shard-index "$i"
    )
    if [[ -n "$LONGCAT_LIMIT" ]]; then
        args+=(--limit "$LONGCAT_LIMIT")
    fi
    echo "[$(date --iso-8601=seconds)] launching shard $i on GPU $i -> $shard_log"
    CUDA_VISIBLE_DEVICES="$i" nohup "${args[@]}" > "$shard_log" 2>&1 &
    pids+=("$!")
done

echo "[$(date --iso-8601=seconds)] launched ${#pids[@]} shards: pids=${pids[*]}"
printf '%s\n' "${pids[@]}" > "$LOG_DIR/shard_pids.txt"

fail=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "[$(date --iso-8601=seconds)] shard $idx (pid ${pids[$idx]}) completed OK"
    else
        rc=$?
        fail=1
        echo "[$(date --iso-8601=seconds)] shard $idx (pid ${pids[$idx]}) FAILED rc=$rc"
    fi
done

echo "[$(date --iso-8601=seconds)] all shards finished. output_dir=$OUTPUT_DIR"
exit "$fail"
