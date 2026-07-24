#!/usr/bin/env bash
# LingBot-Video Dense uses eight one-GPU workers; MoE automatically uses one
# eight-GPU distributed worker.  Both keep the pipeline loaded across samples.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VARIANT="${LINGBOT_VIDEO_VARIANT:-dense}"

case "$VARIANT" in
    dense)
        DEFAULT_MODEL_CONFIG="config/models/lingbot_video_dense.yaml"
        DEFAULT_OUTPUT_NAME="lingbot-video-dense-production"
        ;;
    moe)
        DEFAULT_MODEL_CONFIG="config/models/lingbot_video_moe.yaml"
        DEFAULT_OUTPUT_NAME="lingbot-video-moe-production"
        ;;
    *)
        echo "ERROR: LINGBOT_VIDEO_VARIANT must be dense or moe, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${LINGBOT_VIDEO_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
export WORLDARENA_BATCH_MANIFEST="${LINGBOT_VIDEO_MANIFEST:-${WORLDARENA_BATCH_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl}}"
export WORLDARENA_BATCH_SUITES="${LINGBOT_VIDEO_SUITES:-${WORLDARENA_BATCH_SUITES:-image_static,image_dynamic}}"
export WORLDARENA_BATCH_OUTPUT_DIR="${LINGBOT_VIDEO_OUTPUT_DIR:-${WORLDARENA_BATCH_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/$DEFAULT_OUTPUT_NAME}}"
export WORLDARENA_BATCH_TOTAL_GPUS="${LINGBOT_VIDEO_TOTAL_GPUS:-${WORLDARENA_BATCH_TOTAL_GPUS:-8}}"
export WORLDARENA_BATCH_GPU_OFFSET="${LINGBOT_VIDEO_GPU_OFFSET:-${WORLDARENA_BATCH_GPU_OFFSET:-0}}"
export WORLDARENA_BATCH_LIMIT="${LINGBOT_VIDEO_LIMIT:-${WORLDARENA_BATCH_LIMIT:-}}"
export WORLDARENA_BATCH_DETACH="${LINGBOT_VIDEO_DETACH:-${WORLDARENA_BATCH_DETACH:-1}}"
export WORLDARENA_BATCH_WAIT_FOR_GPUS="${LINGBOT_VIDEO_WAIT_FOR_GPUS:-${WORLDARENA_BATCH_WAIT_FOR_GPUS:-0}}"

exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
