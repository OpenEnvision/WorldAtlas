#!/usr/bin/env bash
# Launch Stable Virtual Camera (SVC / seva) on the WorldArena image_dynamic suite
# using 8 GPUs.
#
# Strategy: 8 independent single-GPU shards. Shard i runs on GPU i and processes
# manifest[i::8] of the image_dynamic samples. The worldarena CLI orchestrates;
# the SVC batch runner (config python_bin) is spawned ONCE per shard and loads the
# 5GB diffusion checkpoint ONCE, then generates every sample in the shard
# sequentially (no per-sample reload).
#
# Usage:
#   bash scripts/launch_stable_virtual_camera_image_dynamic_8gpu.sh
#   SVC_LIMIT=2 bash scripts/launch_stable_virtual_camera_image_dynamic_8gpu.sh   # smoke (per shard limit)
#   SVC_NUM_SHARDS=2 SVC_LIMIT=1 bash scripts/launch_stable_virtual_camera_image_dynamic_8gpu.sh  # tiny smoke
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
# Orchestration python: must be able to `import worldarena` (mirrors the longcat launcher).
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/stable_virtual_camera.yaml"
BENCHMARK_CONFIG="config/benchmark_openenvision.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"

NUM_SHARDS="${SVC_NUM_SHARDS:-8}"
SVC_LIMIT="${SVC_LIMIT:-}"   # optional --limit passed to each shard (smoke testing)

TIMESTAMP="${SVC_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${SVC_OUTPUT_DIR:-$WORLD_ROOT/bench_output/stable-virtual-camera-image-dynamic-5s}"
LOG_DIR="$WORLD_ROOT/artifacts/logs/svc_image_dynamic_8gpu_${TIMESTAMP}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[$(date --iso-8601=seconds)] Stable Virtual Camera image_dynamic 8-GPU launcher started"
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_DIR"
echo "num_shards=$NUM_SHARDS"
echo "limit_per_shard=${SVC_LIMIT:-<none>}"
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
    if [[ -n "$SVC_LIMIT" ]]; then
        args+=(--limit "$SVC_LIMIT")
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
