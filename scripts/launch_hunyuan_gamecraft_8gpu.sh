#!/usr/bin/env bash
# Hunyuan-GameCraft: eight one-GPU checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${HUNYUAN_GAMECRAFT_MODEL_CONFIG:-config/models/hunyuan_gamecraft.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
