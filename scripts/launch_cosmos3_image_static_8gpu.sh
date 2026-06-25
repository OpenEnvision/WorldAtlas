#!/usr/bin/env bash
# Launch Cosmos3-Nano (cosmos-framework) on the WorldArena image_static suite using 8 GPUs.
#
# Strategy: 8 independent single-GPU shards. Shard i runs on GPU i and processes
# manifest[i::8] of the image_static samples. Each shard loads the checkpoint ONCE
# via cosmos_framework.scripts.inference and generates every sample in the shard
# sequentially (single JSONL batch per shard).
#
# Usage:
#   bash scripts/launch_cosmos3_image_static_8gpu.sh
#   COSMOS3_LIMIT=2 bash scripts/launch_cosmos3_image_static_8gpu.sh   # smoke (per shard limit)
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
source "/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/setup_paths.sh"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/cosmos-framework/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/cosmos3_nano.yaml"
BENCHMARK_CONFIG="config/benchmark_openenvision.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"

NUM_SHARDS="${NUM_SHARDS:-8}"
COSMOS3_LIMIT="${COSMOS3_LIMIT:-}"

TIMESTAMP="${COSMOS3_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${COSMOS3_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/cosmos3-nano-image-static-8gpu-${TIMESTAMP}}"
LOG_DIR="$WORLD_ROOT/artifacts/logs/cosmos3_image_static_8gpu_${TIMESTAMP}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date --iso-8601=seconds)] Cosmos3-Nano image_static 8-GPU launcher started"
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_DIR"
echo "num_shards=$NUM_SHARDS"
echo "limit_per_shard=${COSMOS3_LIMIT:-<none>}"
echo "orchestration_python=$PYTHON_BIN"

pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
    shard_log="$LOG_DIR/shard${i}.log"
    args=(
        "$PYTHON_BIN" -u -m worldarena.cli generate
        --config "$BENCHMARK_CONFIG"
        --model-config "$MODEL_CONFIG"
        --manifest "$MANIFEST"
        --suites image_static
        --output-dir "$OUTPUT_DIR"
        --num-shards "$NUM_SHARDS"
        --shard-index "$i"
    )
    if [[ -n "$COSMOS3_LIMIT" ]]; then
        args+=(--limit "$COSMOS3_LIMIT")
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
