#!/usr/bin/env bash
# WorldFM: eight checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${WORLDFM_MODEL_CONFIG:-config/models/worldfm.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
