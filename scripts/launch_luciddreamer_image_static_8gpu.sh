#!/usr/bin/env bash
# Launch LucidDreamer on the WorldArena image_static suite using 8 GPUs.
#
# Strategy: 8 independent single-GPU shards. Each shard starts one WorldArena
# generation process, and each process starts one persistent LucidDreamer runner
# that loads the upstream checkpoints once before processing its shard.
set -euo pipefail

WORLD_ROOT="${WORLD_ROOT:-/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt}"
BENCHMARK_PYTHON_BIN="${BENCHMARK_PYTHON_BIN:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin/python}"
LUCIDDREAMER_PYTHON_BIN="${LUCIDDREAMER_PYTHON_BIN:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/luciddreamer/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-config/models/luciddreamer.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-config/benchmark.yaml}"
MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-8}"
LUCIDDREAMER_LIMIT="${LUCIDDREAMER_LIMIT:-}"
LUCIDDREAMER_OVERWRITE="${LUCIDDREAMER_OVERWRITE:-0}"
TIMESTAMP="${LUCIDDREAMER_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${LUCIDDREAMER_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/luciddreamer-image-static-8gpu-$TIMESTAMP}"
LOG_DIR="${LUCIDDREAMER_LOG_DIR:-$WORLD_ROOT/artifacts/logs/luciddreamer_image_static_8gpu_$TIMESTAMP}"
FFMPEG_BIN_DIR="${FFMPEG_BIN_DIR:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin}"
LUCID_TORCH_LIB="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/luciddreamer/lib/python3.9/site-packages/torch/lib"
LUCID_ENV_LIB="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/luciddreamer/lib"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CKPT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="$LUCID_TORCH_LIB:$LUCID_ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$FFMPEG_BIN_DIR:$PATH"

{
  echo "WORLD_ROOT=$WORLD_ROOT"
  echo "CKPT_ROOT=$CKPT_ROOT"
  echo "BENCHMARK_PYTHON_BIN=$BENCHMARK_PYTHON_BIN"
  echo "LUCIDDREAMER_PYTHON_BIN=$LUCIDDREAMER_PYTHON_BIN"
  echo "MODEL_CONFIG=$MODEL_CONFIG"
  echo "BENCHMARK_CONFIG=$BENCHMARK_CONFIG"
  echo "MANIFEST=$MANIFEST"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "LUCIDDREAMER_LIMIT=${LUCIDDREAMER_LIMIT:-<none>}"
  echo "LUCIDDREAMER_OVERWRITE=$LUCIDDREAMER_OVERWRITE"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "FFMPEG_BIN_DIR=$FFMPEG_BIN_DIR"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "PATH=$PATH"
  echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
} > "$LOG_DIR/run_env.txt"

echo "[$(date --iso-8601=seconds)] LucidDreamer image_static ${NUM_SHARDS}-GPU generation started"
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_DIR"

pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  shard_log="$LOG_DIR/shard${i}.log"
  args=(
    "$BENCHMARK_PYTHON_BIN" -u -m worldarena.cli generate
    --config "$BENCHMARK_CONFIG"
    --model-config "$MODEL_CONFIG"
    --manifest "$MANIFEST"
    --suites image_static
    --output-dir "$OUTPUT_DIR"
    --num-shards "$NUM_SHARDS"
    --shard-index "$i"
  )
  if [[ -n "$LUCIDDREAMER_LIMIT" ]]; then
    args+=(--limit "$LUCIDDREAMER_LIMIT")
  fi
  if [[ "$LUCIDDREAMER_OVERWRITE" == "1" ]]; then
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
