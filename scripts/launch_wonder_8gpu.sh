#!/usr/bin/env bash
# WonderJourney / WonderWorld: eight checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${WONDER_VARIANT:-journey}"
case "$VARIANT" in
    journey|wonderjourney)
        DEFAULT_MODEL_CONFIG="config/models/wonderjourney.yaml"
        ;;
    world|wonderworld)
        DEFAULT_MODEL_CONFIG="config/models/wonderworld.yaml"
        ;;
    *)
        echo "ERROR: WONDER_VARIANT must be journey or world, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${WONDER_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
