#!/usr/bin/env bash
# Open Dreamer: eight checkpoint-resident workers, one JAX process per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${OPEN_DREAMER_MODEL_CONFIG:-config/models/open_dreamer.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
