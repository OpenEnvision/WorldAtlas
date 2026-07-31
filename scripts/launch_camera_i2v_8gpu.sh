#!/usr/bin/env bash
# CamI2V / RealCam-I2V: eight one-GPU checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ARENA_ROOT="$(cd -- "$WORLD_ROOT/.." && pwd)"
VARIANT="${CAMERA_I2V_MODEL:-cami2v}"

# CamI2V / RealCam-I2V weights live under ../world_models; expose them as an
# approved checkpoint root so symlinked ckpt/ paths pass validation.
export WORLDARENA_CHECKPOINT_ROOT="${WORLDARENA_CHECKPOINT_ROOT:-$ARENA_ROOT/world_models}"

case "$VARIANT" in
    cami2v)
        DEFAULT_MODEL_CONFIG="config/models/cami2v.yaml"
        DEFAULT_OUTPUT_NAME="cami2v-production"
        ;;
    realcam_i2v|realcam-i2v|realcami2v)
        DEFAULT_MODEL_CONFIG="config/models/realcam_i2v.yaml"
        DEFAULT_OUTPUT_NAME="realcam-i2v-production"
        ;;
    *)
        echo "ERROR: CAMERA_I2V_MODEL must be cami2v or realcam_i2v, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${CAMERA_I2V_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
export WORLDARENA_BATCH_MANIFEST="${CAMERA_I2V_MANIFEST:-${WORLDARENA_BATCH_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl}}"
export WORLDARENA_BATCH_SUITES="${CAMERA_I2V_SUITES:-${CAMERA_I2V_SUITE:-${WORLDARENA_BATCH_SUITES:-image_static,image_dynamic}}}"
export WORLDARENA_BATCH_OUTPUT_DIR="${CAMERA_I2V_OUTPUT_DIR:-${WORLDARENA_BATCH_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/$DEFAULT_OUTPUT_NAME}}"
export WORLDARENA_BATCH_TOTAL_GPUS="${CAMERA_I2V_TOTAL_GPUS:-${CAMERA_I2V_NUM_SHARDS:-${WORLDARENA_BATCH_TOTAL_GPUS:-8}}}"
export WORLDARENA_BATCH_GPU_OFFSET="${CAMERA_I2V_GPU_OFFSET:-${WORLDARENA_BATCH_GPU_OFFSET:-0}}"
export WORLDARENA_BATCH_LIMIT="${CAMERA_I2V_LIMIT:-${WORLDARENA_BATCH_LIMIT:-}}"
export WORLDARENA_BATCH_DETACH="${CAMERA_I2V_DETACH:-${WORLDARENA_BATCH_DETACH:-1}}"
export WORLDARENA_BATCH_WAIT_FOR_GPUS="${CAMERA_I2V_WAIT_FOR_GPUS:-${WORLDARENA_BATCH_WAIT_FOR_GPUS:-0}}"

exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
