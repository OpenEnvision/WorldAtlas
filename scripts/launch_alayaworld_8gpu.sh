#!/usr/bin/env bash
# Run AlayaWorld production with one independent WorldArena shard per GPU.
# Defaults to the complete local 5,000-sample image manifest and 8 A100s.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${ALAYAWORLD_PYTHON_BIN:-$WORLD_ROOT/../conda/envs/alayaworld/bin/python}"
BENCHMARK_CONFIG="${ALAYAWORLD_BENCHMARK_CONFIG:-config/benchmark.yaml}"
LOCAL_MODEL_CONFIG="config/models/alayaworld_production_local.yaml"
if [[ -f "$WORLD_ROOT/$LOCAL_MODEL_CONFIG" ]]; then
    DEFAULT_MODEL_CONFIG="$LOCAL_MODEL_CONFIG"
else
    DEFAULT_MODEL_CONFIG="config/models/alayaworld.yaml"
fi
MODEL_CONFIG="${ALAYAWORLD_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
MANIFEST="${ALAYAWORLD_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl}"
SUITES="${ALAYAWORLD_SUITES:-image_static,image_dynamic}"
NUM_SHARDS="${ALAYAWORLD_NUM_SHARDS:-8}"
GPU_OFFSET="${ALAYAWORLD_GPU_OFFSET:-0}"
LIMIT="${ALAYAWORLD_LIMIT:-}"
DETACH="${ALAYAWORLD_DETACH:-1}"

TIMESTAMP="${ALAYAWORLD_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

export WORLDARENA_BATCH_PYTHON_BIN="$PYTHON_BIN"
export WORLDARENA_BATCH_BENCHMARK_CONFIG="$BENCHMARK_CONFIG"
export WORLDARENA_BATCH_MODEL_CONFIG="$MODEL_CONFIG"
export WORLDARENA_BATCH_MANIFEST="$MANIFEST"
export WORLDARENA_BATCH_SUITES="$SUITES"
export WORLDARENA_BATCH_TOTAL_GPUS="$NUM_SHARDS"
export WORLDARENA_BATCH_GPU_OFFSET="$GPU_OFFSET"
export WORLDARENA_BATCH_LIMIT="$LIMIT"
export WORLDARENA_BATCH_DETACH="$DETACH"
export WORLDARENA_BATCH_FAIL_FAST="${ALAYAWORLD_FAIL_FAST:-${LIMIT:+1}}"
export WORLDARENA_BATCH_TIMESTAMP="$TIMESTAMP"
export WORLDARENA_BATCH_OUTPUT_DIR="${ALAYAWORLD_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/alayaworld-production}"
export WORLDARENA_BATCH_LOG_DIR="${ALAYAWORLD_LOG_DIR:-$WORLD_ROOT/artifacts/logs/alayaworld_production_$TIMESTAMP}"
export WORLDARENA_BATCH_WAIT_FOR_GPUS="${ALAYAWORLD_WAIT_FOR_GPUS:-${WORLDARENA_BATCH_WAIT_FOR_GPUS:-0}}"

exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
