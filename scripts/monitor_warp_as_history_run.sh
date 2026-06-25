#!/usr/bin/env bash
set -euo pipefail

: "${MASTER_PID:?MASTER_PID is required}"
: "${WORLD_ROOT:?WORLD_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"

while kill -0 "$MASTER_PID" 2>/dev/null; do
  sleep "${POLL_SECONDS:-300}"
done

cd "$WORLD_ROOT"
set +e
"$PYTHON_BIN" scripts/verify_warp_as_history_image_static_run.py \
  --manifest artifacts/benchmark/manifest/worldarena_manifest.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --duration-samples "${DURATION_SAMPLES:-32}" \
  --strict > "$LOG_DIR/final_verification.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$LOG_DIR/final_verification.exit"
exit "$status"
