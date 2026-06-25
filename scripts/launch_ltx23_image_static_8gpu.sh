#!/usr/bin/env bash
# Launch LTX-2.3 on the WorldArena image_static suite using 8 GPUs.
#
# Strategy: 8 independent single-GPU shards. Each shard starts one long-lived
# WorldArena generation process; the LTX batch runner installs a process-local
# weight cache so each unique checkpoint-backed module is loaded once, then
# reused for every sample assigned to that shard.
#
# Smoke test:
#   LTX23_LIMIT=8 bash scripts/launch_ltx23_image_static_8gpu.sh
#
# Full run:
#   bash scripts/launch_ltx23_image_static_8gpu.sh
set -euo pipefail

WORLD_ROOT="${WORLD_ROOT:-/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/yangboxue/visual_generation/juanxi/github_repos/ltx2.3/.venv/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-artifacts/run_configs/ltx23_juanxi_image_static_5s.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-config/benchmark.yaml}"
MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-8}"
LTX23_LIMIT="${LTX23_LIMIT:-}"
LTX23_OVERWRITE="${LTX23_OVERWRITE:-0}"
TIMESTAMP="${LTX23_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${LTX23_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/ltx23-image-static-8gpu-$TIMESTAMP}"
LOG_DIR="${LTX23_LOG_DIR:-$WORLD_ROOT/artifacts/logs/ltx23_image_static_8gpu_$TIMESTAMP}"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CKPT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "WORLD_ROOT=$WORLD_ROOT"
  echo "CKPT_ROOT=$CKPT_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "MODEL_CONFIG=$MODEL_CONFIG"
  echo "BENCHMARK_CONFIG=$BENCHMARK_CONFIG"
  echo "MANIFEST=$MANIFEST"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "LTX23_LIMIT=${LTX23_LIMIT:-<none>}"
  echo "LTX23_OVERWRITE=$LTX23_OVERWRITE"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
} > "$LOG_DIR/run_env.txt"

echo "[$(date --iso-8601=seconds)] LTX-2.3 image_static ${NUM_SHARDS}-GPU generation started"
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
  if [[ -n "$LTX23_LIMIT" ]]; then
    args+=(--limit "$LTX23_LIMIT")
  fi
  if [[ "$LTX23_OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
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
