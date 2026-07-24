#!/usr/bin/env bash
# Yume-1.5: eight checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${YUME_MODEL_CONFIG:-config/models/yume.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
