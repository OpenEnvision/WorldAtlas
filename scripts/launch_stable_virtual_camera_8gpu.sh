#!/usr/bin/env bash
# Stable Virtual Camera: eight checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${STABLE_VIRTUAL_CAMERA_MODEL_CONFIG:-config/models/stable_virtual_camera.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
