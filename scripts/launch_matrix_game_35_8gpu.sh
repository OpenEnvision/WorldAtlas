#!/usr/bin/env bash
# Matrix-Game 3.5 basic-metric generation on 8 GPUs.
#
# Unlike the load-once resident models, Matrix-Game 3.5's adapter reports
# checkpoint_policy='reload_per_sample', so it CANNOT use launch_model_batch_8gpu.sh
# (that launcher enforces --require-load-once). Instead we run the native
# `worldarena generate` sharding: one process per GPU, each owning one manifest
# shard (--num-shards 8 --shard-index i).
#
# Select camera contract via MATRIX_GAME_35_PERSON=first|third.
#
# Usage:
#   bash scripts/launch_matrix_game_35_8gpu.sh
#   MATRIX_GAME_35_PERSON=third MATRIX_GAME_35_LIMIT=2 bash scripts/launch_matrix_game_35_8gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ARENA_ROOT="$(cd -- "$WORLD_ROOT/.." && pwd)"
PERSON="${MATRIX_GAME_35_PERSON:-first}"

case "$PERSON" in
    first|first-person|first_person)
        MODEL_CONFIG="${MATRIX_GAME_35_MODEL_CONFIG:-config/models/matrix_game_35_first.yaml}"
        OUTPUT_NAME="matrix-game-3.5-first-production"
        ;;
    third|third-person|third_person)
        MODEL_CONFIG="${MATRIX_GAME_35_MODEL_CONFIG:-config/models/matrix_game_35_third.yaml}"
        OUTPUT_NAME="matrix-game-3.5-third-production"
        ;;
    *)
        echo "ERROR: MATRIX_GAME_35_PERSON must be first or third, got: $PERSON" >&2
        exit 2
        ;;
esac

BENCHMARK_CONFIG="${MATRIX_GAME_35_BENCHMARK_CONFIG:-config/benchmark.yaml}"
MANIFEST="${MATRIX_GAME_35_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
SUITES="${MATRIX_GAME_35_SUITES:-image_static,image_dynamic}"
OUTPUT_DIR="${MATRIX_GAME_35_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/$OUTPUT_NAME}"
TOTAL_GPUS="${MATRIX_GAME_35_TOTAL_GPUS:-8}"
GPU_OFFSET="${MATRIX_GAME_35_GPU_OFFSET:-0}"
LIMIT="${MATRIX_GAME_35_LIMIT:-}"
OVERWRITE="${MATRIX_GAME_35_OVERWRITE:-0}"
DETACH="${MATRIX_GAME_35_DETACH:-1}"

# The generate orchestrator needs the worldarena runtime deps; the Matrix-Game
# subprocess python comes from the model config (thirdparty checkpoints etc.).
DEFAULT_PYTHON="$ARENA_ROOT/envs/alayaworld/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${MATRIX_GAME_35_PYTHON_BIN:-$DEFAULT_PYTHON}"

# Matrix-Game weights live under ../world_models; expose them as an approved
# checkpoint root so ckpt/ symlinks resolve past the ckpt-root guard.
export WORLDARENA_CHECKPOINT_ROOT="${WORLDARENA_CHECKPOINT_ROOT:-$ARENA_ROOT/world_models}"
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORLD_ROOT"
for required in "$BENCHMARK_CONFIG" "$MODEL_CONFIG" "$MANIFEST"; do
    [[ -e "$required" ]] || { echo "ERROR: required path missing: $required" >&2; exit 2; }
done

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${MATRIX_GAME_35_LOG_DIR:-$WORLD_ROOT/artifacts/logs/matrix_game_35_${PERSON}_$TIMESTAMP}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date --iso-8601=seconds)] Matrix-Game 3.5 ($PERSON) native 8-shard generation"
echo "model_config=$MODEL_CONFIG manifest=$MANIFEST suites=$SUITES"
echo "output_dir=$OUTPUT_DIR log_dir=$LOG_DIR total_gpus=$TOTAL_GPUS limit=${LIMIT:-<none>}"

pids=()
for ((shard=0; shard<TOTAL_GPUS; shard++)); do
    gpu=$((GPU_OFFSET + shard))
    shard_log="$LOG_DIR/shard${shard}.log"
    gen_args=(
        "$PYTHON_BIN" -u -m worldarena.cli generate
        --config "$BENCHMARK_CONFIG"
        --model-config "$MODEL_CONFIG"
        --manifest "$MANIFEST"
        --suites "$SUITES"
        --output-dir "$OUTPUT_DIR"
        --num-shards "$TOTAL_GPUS"
        --shard-index "$shard"
    )
    [[ -n "$LIMIT" ]] && gen_args+=(--limit "$LIMIT")
    [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]] && gen_args+=(--overwrite)
    echo "[$(date --iso-8601=seconds)] shard=$shard gpu=$gpu log=$shard_log"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "${gen_args[@]}" >"$shard_log" 2>&1 &
    pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$LOG_DIR/shard_pids.txt"
echo "[$(date --iso-8601=seconds)] launched pids=${pids[*]}"

if [[ "$DETACH" == "1" || "$DETACH" == "true" ]]; then
    echo "[$(date --iso-8601=seconds)] detached; logs=$LOG_DIR"
    exit 0
fi

failed=0
for ((shard=0; shard<TOTAL_GPUS; shard++)); do
    if wait "${pids[$shard]}"; then
        echo "[$(date --iso-8601=seconds)] shard $shard OK"
    else
        failed=1
        echo "[$(date --iso-8601=seconds)] shard $shard FAILED" >&2
    fi
done
echo "[$(date --iso-8601=seconds)] all shards finished; output_dir=$OUTPUT_DIR"
exit "$failed"
