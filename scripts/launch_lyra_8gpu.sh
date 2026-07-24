#!/usr/bin/env bash
# Lyra-1 / Lyra-2: eight checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${LYRA_VARIANT:-1}"
case "$VARIANT" in
    1|lyra1)
        DEFAULT_MODEL_CONFIG="config/models/lyra1.yaml"
        ;;
    2|lyra2)
        DEFAULT_MODEL_CONFIG="config/models/lyra2.yaml"
        ;;
    *)
        echo "ERROR: LYRA_VARIANT must be 1 or 2, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${LYRA_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
