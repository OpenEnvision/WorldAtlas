#!/usr/bin/env bash
# LingBot-World variants use one persistent distributed worker across 8 GPUs
# when nproc_per_node=8, and never reload the checkpoint between samples.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VARIANT="${LINGBOT_WORLD_VARIANT:-base}"

case "$VARIANT" in
    base)
        DEFAULT_MODEL_CONFIG="config/models/lingbot_world.yaml"
        DEFAULT_OUTPUT_NAME="lingbot-world-production"
        ;;
    fast)
        DEFAULT_MODEL_CONFIG="config/models/lingbot_world_fast.yaml"
        DEFAULT_OUTPUT_NAME="lingbot-world-fast-production"
        ;;
    act|base_act|base-act)
        DEFAULT_MODEL_CONFIG="config/models/lingbot_world_base_act_preview.yaml"
        DEFAULT_OUTPUT_NAME="lingbot-world-base-act-production"
        ;;
    *)
        echo "ERROR: LINGBOT_WORLD_VARIANT must be base, fast, or act, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${LINGBOT_WORLD_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
export WORLDARENA_BATCH_MANIFEST="${LINGBOT_WORLD_MANIFEST:-${WORLDARENA_BATCH_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl}}"
export WORLDARENA_BATCH_SUITES="${LINGBOT_WORLD_SUITES:-${WORLDARENA_BATCH_SUITES:-image_static,image_dynamic}}"
export WORLDARENA_BATCH_OUTPUT_DIR="${LINGBOT_WORLD_OUTPUT_DIR:-${WORLDARENA_BATCH_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/$DEFAULT_OUTPUT_NAME}}"
export WORLDARENA_BATCH_TOTAL_GPUS="${LINGBOT_WORLD_TOTAL_GPUS:-${WORLDARENA_BATCH_TOTAL_GPUS:-8}}"
export WORLDARENA_BATCH_GPU_OFFSET="${LINGBOT_WORLD_GPU_OFFSET:-${WORLDARENA_BATCH_GPU_OFFSET:-0}}"
export WORLDARENA_BATCH_LIMIT="${LINGBOT_WORLD_LIMIT:-${WORLDARENA_BATCH_LIMIT:-}}"
export WORLDARENA_BATCH_DETACH="${LINGBOT_WORLD_DETACH:-${WORLDARENA_BATCH_DETACH:-1}}"
export WORLDARENA_BATCH_WAIT_FOR_GPUS="${LINGBOT_WORLD_WAIT_FOR_GPUS:-${WORLDARENA_BATCH_WAIT_FOR_GPUS:-0}}"

exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
