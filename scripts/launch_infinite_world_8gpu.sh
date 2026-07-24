#!/usr/bin/env bash
# InfiniteWorld: eight one-GPU checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORLDARENA_BATCH_MODEL_CONFIG="${INFINITE_WORLD_MODEL_CONFIG:-config/models/infinite_world.yaml}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
