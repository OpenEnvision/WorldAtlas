#!/usr/bin/env bash
# Launch AC3D on the WorldArena image_static suite using 8 single-GPU shards.
#
# Each shard owns manifest[i::8] and keeps one AC3D pipeline resident while it
# generates its shard. Use AC3D_LIMIT for a smoke run before full generation.
#
# Usage:
#   bash scripts/launch_ac3d_image_static_8gpu.sh
#   AC3D_LIMIT=1 bash scripts/launch_ac3d_image_static_8gpu.sh
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/ac3d/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/ac3d.yaml"
BENCHMARK_CONFIG="config/benchmark_openenvision.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"

NUM_SHARDS="${NUM_SHARDS:-8}"
AC3D_LIMIT="${AC3D_LIMIT:-}"

TIMESTAMP="${AC3D_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${AC3D_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/ac3d-image-static-8gpu-singleload-${TIMESTAMP}}"
LOG_DIR="$WORLD_ROOT/artifacts/logs/ac3d_image_static_8gpu_${TIMESTAMP}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date --iso-8601=seconds)] AC3D image_static 8-GPU launcher started"
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_DIR"
echo "num_shards=$NUM_SHARDS"
echo "limit_per_shard=${AC3D_LIMIT:-<none>}"
echo "python=$PYTHON_BIN"

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
    if [[ -n "$AC3D_LIMIT" ]]; then
        args+=(--limit "$AC3D_LIMIT")
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
