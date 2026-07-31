#!/usr/bin/env bash
# LingBot-World v2 reuses the v1 (family: lingbot_world) adapter and runs as one
# persistent distributed worker across 8 GPUs (nproc_per_node=8), never reloading
# the checkpoint between samples.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

export WORLDARENA_BATCH_MODEL_CONFIG="${LINGBOT_WORLD_V2_MODEL_CONFIG:-config/models/lingbot_world_v2.yaml}"
export WORLDARENA_BATCH_MANIFEST="${LINGBOT_WORLD_V2_MANIFEST:-${WORLDARENA_BATCH_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl}}"
export WORLDARENA_BATCH_SUITES="${LINGBOT_WORLD_V2_SUITES:-${WORLDARENA_BATCH_SUITES:-image_static,image_dynamic}}"
export WORLDARENA_BATCH_OUTPUT_DIR="${LINGBOT_WORLD_V2_OUTPUT_DIR:-${WORLDARENA_BATCH_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/lingbot-world-v2-production}}"
export WORLDARENA_BATCH_TOTAL_GPUS="${LINGBOT_WORLD_V2_TOTAL_GPUS:-${WORLDARENA_BATCH_TOTAL_GPUS:-8}}"
export WORLDARENA_BATCH_GPU_OFFSET="${LINGBOT_WORLD_V2_GPU_OFFSET:-${WORLDARENA_BATCH_GPU_OFFSET:-0}}"
export WORLDARENA_BATCH_LIMIT="${LINGBOT_WORLD_V2_LIMIT:-${WORLDARENA_BATCH_LIMIT:-}}"
export WORLDARENA_BATCH_DETACH="${LINGBOT_WORLD_V2_DETACH:-${WORLDARENA_BATCH_DETACH:-1}}"
export WORLDARENA_BATCH_WAIT_FOR_GPUS="${LINGBOT_WORLD_V2_WAIT_FOR_GPUS:-${WORLDARENA_BATCH_WAIT_FOR_GPUS:-0}}"

exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
