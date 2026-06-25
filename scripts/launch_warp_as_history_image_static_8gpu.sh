#!/usr/bin/env bash
# Launch Warp-as-History on the WorldArena image_static suite using 8 GPUs.
# Each shard starts one persistent batch runner, loads the checkpoint once, and
# then generates every sample assigned to that GPU.
set -euo pipefail

WORLD_ROOT="${WORLD_ROOT:-/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/warp-as-history/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-config/models/warp_as_history.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-config/benchmark.yaml}"
MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-8}"
TIMESTAMP="${WAH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${WAH_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/warp-as-history-image-static-8gpu-$TIMESTAMP}"
LOG_DIR="${WAH_LOG_DIR:-$WORLD_ROOT/artifacts/logs/warp_as_history_image_static_8gpu_$TIMESTAMP}"
WAH_LIMIT="${WAH_LIMIT:-}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CKPT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

{
  echo "WORLD_ROOT=$WORLD_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "CKPT_ROOT=$CKPT_ROOT"
  echo "MODEL_CONFIG=$MODEL_CONFIG"
  echo "BENCHMARK_CONFIG=$BENCHMARK_CONFIG"
  echo "MANIFEST=$MANIFEST"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "WAH_LIMIT=${WAH_LIMIT:-<none>}"
} > "$LOG_DIR/run_env.txt"

echo "[$(date --iso-8601=seconds)] Warp-as-History image_static ${NUM_SHARDS}-GPU generation started"
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_DIR"

pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
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
  if [[ -n "$WAH_LIMIT" ]]; then
    args+=(--limit "$WAH_LIMIT")
  fi
  echo "[$(date --iso-8601=seconds)] launching shard $i on GPU $i -> $shard_log"
  CUDA_VISIBLE_DEVICES="$i" "${args[@]}" > "$shard_log" 2>&1 &
  pids+=("$!")
done

printf '%s\n' "${pids[@]}" > "$LOG_DIR/shard_pids.txt"
echo "[$(date --iso-8601=seconds)] launched pids=${pids[*]}"

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[$(date --iso-8601=seconds)] shard $idx completed OK"
  else
    rc=$?
    fail=1
    echo "[$(date --iso-8601=seconds)] shard $idx FAILED rc=$rc"
  fi
done

echo "[$(date --iso-8601=seconds)] all shards finished fail=$fail output_dir=$OUTPUT_DIR"
exit "$fail"
