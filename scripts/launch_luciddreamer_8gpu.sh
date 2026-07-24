#!/usr/bin/env bash
# LucidDreamer: eight one-GPU checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${LUCIDDREAMER_MODEL_CONFIG:-config/models/luciddreamer.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
