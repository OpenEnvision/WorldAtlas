#!/usr/bin/env bash
# Cosmos-Predict2.5 2B / 14B: one persistent distributed worker on eight GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${COSMOS_PREDICT25_VARIANT:-2b}"
case "$VARIANT" in
    2b)
        DEFAULT_MODEL_CONFIG="config/models/cosmos_predict25_2b.yaml"
        ;;
    14b)
        DEFAULT_MODEL_CONFIG="config/models/cosmos_predict25_14b.yaml"
        ;;
    *)
        echo "ERROR: COSMOS_PREDICT25_VARIANT must be 2b or 14b, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${COSMOS_PREDICT25_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
