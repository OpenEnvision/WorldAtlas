#!/usr/bin/env bash
# FlashWorld video / image variants: eight checkpoint-resident workers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${FLASHWORLD_VARIANT:-video}"
case "$VARIANT" in
    video|video_static)
        DEFAULT_MODEL_CONFIG="config/models/flashworld.yaml"
        ;;
    image|image_static)
        DEFAULT_MODEL_CONFIG="config/models/flashworld_image_static.yaml"
        ;;
    *)
        echo "ERROR: FLASHWORLD_VARIANT must be video or image, got: $VARIANT" >&2
        exit 2
        ;;
esac

export WORLDARENA_BATCH_MODEL_CONFIG="${FLASHWORLD_MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
exec "$SCRIPT_DIR/launch_model_batch_8gpu.sh"
