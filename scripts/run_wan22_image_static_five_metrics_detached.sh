#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${WORLDARENA_PYTHON:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldarena/bin/python}"
CONFIG="${WORLDARENA_CONFIG:-config/benchmark.yaml}"
MANIFEST="${WORLDARENA_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
PREDICTIONS_ROOT="${WORLDARENA_PREDICTIONS_ROOT:-artifacts/predictions/wan22-ti2v-5b-image-static-plan-missing}"
MODEL_NAME="${WORLDARENA_MODEL_NAME:-wan22-ti2v-5b-image-static-plan-missing}"
EXPECTED="${WORLDARENA_EXPECTED:-3372}"
MAX_GAP_ATTEMPTS="${WORLDARENA_MAX_GAP_ATTEMPTS:-10}"

RUN_ID="wan22_image_static_five_metrics_20260725"
STATE_DIR="$ROOT/artifacts/benchmark/background_jobs/$RUN_ID"
MANIFEST_DIR="$ROOT/artifacts/benchmark/manifests/$RUN_ID"
REPORT_DIR="$ROOT/artifacts/benchmark/reports"
LOCK_FILE="$STATE_DIR/supervisor.lock"
PID_FILE="$STATE_DIR/supervisor.pid"
STATUS_FILE="$STATE_DIR/status"
DONE_FILE="$STATE_DIR/DONE"

OPTICAL_METRICS="optical_flow_aepe,physical_fps_error,physical_fps_pct_error,physical_fps_intra_video_cv"
CAMERA_METRICS="camera_error"
ALL_METRICS="camera_error,optical_flow_aepe,physical_fps_error,physical_fps_pct_error,physical_fps_intra_video_cv"

OPTICAL_FULL="$REPORT_DIR/wan22-ti2v-5b-image-static-optical-physical-full-24shard-8gpu-20260725"
CAMERA_FULL="$REPORT_DIR/wan22-ti2v-5b-image-static-camera-full-64shard-8gpu-20260725"
OPTICAL_AUDITED="$REPORT_DIR/wan22-ti2v-5b-image-static-optical-physical-audited-20260725"
CAMERA_AUDITED="$REPORT_DIR/wan22-ti2v-5b-image-static-camera-audited-20260725"
FINAL_AUDITED="$REPORT_DIR/wan22-ti2v-5b-image-static-five-metrics-audited-20260725"

mkdir -p "$STATE_DIR" "$MANIFEST_DIR" "$REPORT_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another $RUN_ID supervisor already holds $LOCK_FILE" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

printf '%s\n' "$$" >"$PID_FILE"
rm -f "$DONE_FILE"
SUCCESS=0

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

write_status() {
  local stage="$1"
  local detail="${2:-}"
  local temporary="$STATUS_FILE.tmp"
  {
    printf 'pid=%s\n' "$$"
    printf 'updated_at=%s\n' "$(timestamp)"
    printf 'stage=%s\n' "$stage"
    printf 'detail=%s\n' "$detail"
    printf 'final_report=%s\n' "$FINAL_AUDITED"
  } >"$temporary"
  mv "$temporary" "$STATUS_FILE"
}

on_exit() {
  local exit_code=$?
  if (( SUCCESS == 0 )); then
    write_status "failed" "exit_code=$exit_code"
    log "Supervisor failed with exit code $exit_code"
  fi
}
trap on_exit EXIT

gpu_csv() {
  local repeats="$1"
  local values=()
  local repeat
  local gpu
  for ((repeat = 0; repeat < repeats; repeat += 1)); do
    for gpu in {0..7}; do
      values+=("$gpu")
    done
  done
  local IFS=,
  printf '%s\n' "${values[*]}"
}

OPTICAL_FULL_GPUS="$(gpu_csv 3)"
CAMERA_FULL_GPUS="$(gpu_csv 8)"
GAP_GPUS="$(gpu_csv 1)"

run_sharded() {
  local label="$1"
  local metrics="$2"
  local manifest="$3"
  local output_dir="$4"
  local gpus="$5"
  local exit_code

  write_status "$label" "manifest=$manifest output=$output_dir"
  log "START $label"
  log "metrics=$metrics"
  log "manifest=$manifest"
  log "output=$output_dir"
  set +e
  "$PYTHON_BIN" scripts/run_eval_shards.py \
    --config "$CONFIG" \
    --manifest "$manifest" \
    --predictions-root "$PREDICTIONS_ROOT" \
    --output-dir "$output_dir" \
    --suites image_static \
    --metrics "$metrics" \
    --model-name "$MODEL_NAME" \
    --gpus "$gpus" \
    --max-concurrent-shards 8 \
    --max-concurrent-per-gpu 1 \
    --heartbeat-interval 30
  exit_code=$?
  set -e
  log "END $label exit_code=$exit_code"
}

SOURCE_REPORTS=()

append_output_reports() {
  local output_dir="$1"
  local score_path
  if [[ -s "$output_dir/per_sample_scores.jsonl" ]]; then
    SOURCE_REPORTS+=("$output_dir")
  fi
  while IFS= read -r -d '' score_path; do
    SOURCE_REPORTS+=("$(dirname "$score_path")")
  done < <(
    find "$output_dir" -mindepth 2 -maxdepth 2 -type f \
      -name per_sample_scores.jsonl -size +0c -print0 2>/dev/null | sort -z
  )
}

collect_group_reports() {
  local group="$1"
  local completed_attempts="$2"
  local attempt
  SOURCE_REPORTS=()
  if [[ "$group" == "optical" ]]; then
    append_output_reports "$OPTICAL_FULL"
    for ((attempt = 1; attempt <= completed_attempts; attempt += 1)); do
      append_output_reports "$REPORT_DIR/wan22-ti2v-5b-image-static-optical-physical-gap-r${attempt}-8gpu-20260725"
    done
  else
    append_output_reports "$CAMERA_FULL"
    for ((attempt = 1; attempt <= completed_attempts; attempt += 1)); do
      append_output_reports "$REPORT_DIR/wan22-ti2v-5b-image-static-camera-gap-r${attempt}-8gpu-20260725"
    done
  fi
}

build_gap() {
  local metrics="$1"
  local output="$2"
  local args=(
    scripts/build_metric_gap_manifest.py
    --manifest "$MANIFEST"
    --suite image_static
    --metrics "$metrics"
    --output "$output"
  )
  local report
  for report in "${SOURCE_REPORTS[@]}"; do
    args+=(--report "$report")
  done
  "$PYTHON_BIN" "${args[@]}"
}

gap_count() {
  awk 'END { print NR + 0 }' "$1"
}

merge_group() {
  local metrics="$1"
  local output_dir="$2"
  local args=(
    scripts/merge_metric_reports.py
    --manifest "$MANIFEST"
    --suite image_static
    --metrics "$metrics"
    --predictions-root "$PREDICTIONS_ROOT"
    --model-name "$MODEL_NAME"
    --output-dir "$output_dir"
    --expected "$EXPECTED"
  )
  local report
  if ((${#SOURCE_REPORTS[@]} == 0)); then
    log "No source reports are available for $output_dir"
    return 1
  fi
  for report in "${SOURCE_REPORTS[@]}"; do
    args+=(--report "$report")
  done
  "$PYTHON_BIN" "${args[@]}"
}

complete_group() {
  local group="$1"
  local metrics="$2"
  local full_output="$3"
  local full_gpus="$4"
  local audited_output="$5"
  local attempt
  local gap_manifest
  local count
  local retry_output

  run_sharded "${group}_full" "$metrics" "$MANIFEST" "$full_output" "$full_gpus"

  for ((attempt = 1; attempt <= MAX_GAP_ATTEMPTS; attempt += 1)); do
    collect_group_reports "$group" "$((attempt - 1))"
    gap_manifest="$MANIFEST_DIR/${group}_gap_r${attempt}.jsonl"
    build_gap "$metrics" "$gap_manifest"
    count="$(gap_count "$gap_manifest")"
    log "$group gap attempt=$attempt count=$count"
    if (( count == 0 )); then
      break
    fi
    retry_output="$REPORT_DIR/wan22-ti2v-5b-image-static-${group}-gap-r${attempt}-8gpu-20260725"
    if [[ "$group" == "optical" ]]; then
      retry_output="$REPORT_DIR/wan22-ti2v-5b-image-static-optical-physical-gap-r${attempt}-8gpu-20260725"
    fi
    run_sharded "${group}_gap_r${attempt}" "$metrics" "$gap_manifest" "$retry_output" "$GAP_GPUS"
  done

  collect_group_reports "$group" "$MAX_GAP_ATTEMPTS"
  gap_manifest="$MANIFEST_DIR/${group}_gap_final.jsonl"
  build_gap "$metrics" "$gap_manifest"
  count="$(gap_count "$gap_manifest")"
  if (( count != 0 )); then
    log "$group remains incomplete after $MAX_GAP_ATTEMPTS retries: gap=$count"
    return 1
  fi

  write_status "${group}_merge" "output=$audited_output"
  merge_group "$metrics" "$audited_output"
  log "$group coverage audited: $EXPECTED samples"
}

write_status "starting" "sequential optical/physical then camera"
log "Supervisor pid=$$ root=$ROOT"
log "Eight GPUs are used with one metric worker per GPU"

complete_group "optical" "$OPTICAL_METRICS" "$OPTICAL_FULL" "$OPTICAL_FULL_GPUS" "$OPTICAL_AUDITED"
complete_group "camera" "$CAMERA_METRICS" "$CAMERA_FULL" "$CAMERA_FULL_GPUS" "$CAMERA_AUDITED"

write_status "final_merge" "output=$FINAL_AUDITED"
"$PYTHON_BIN" scripts/merge_metric_reports.py \
  --manifest "$MANIFEST" \
  --suite image_static \
  --metrics "$ALL_METRICS" \
  --report "$OPTICAL_AUDITED" \
  --report "$CAMERA_AUDITED" \
  --predictions-root "$PREDICTIONS_ROOT" \
  --model-name "$MODEL_NAME" \
  --output-dir "$FINAL_AUDITED" \
  --expected "$EXPECTED"

"$PYTHON_BIN" - "$FINAL_AUDITED/run_manifest.json" "$EXPECTED" "$ALL_METRICS" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
expected = int(sys.argv[2])
metrics = tuple(sys.argv[3].split(","))
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("items") != expected:
    raise SystemExit(f"unexpected item count: {payload.get('items')} != {expected}")
coverage = payload.get("metric_coverage") or {}
for metric in metrics:
    metric_coverage = coverage.get(metric) or {}
    if metric_coverage.get("nonnull") != expected:
        raise SystemExit(f"incomplete metric coverage: {metric}={metric_coverage}")
if payload.get("reported_failures"):
    raise SystemExit(f"unexpected failures: {payload['reported_failures']}")
print(json.dumps({"items": expected, "metrics": metrics, "coverage": coverage}, indent=2))
PY

SUCCESS=1
write_status "complete" "items=$EXPECTED metrics=5"
printf 'completed_at=%s\nfinal_report=%s\n' "$(timestamp)" "$FINAL_AUDITED" >"$DONE_FILE"
log "COMPLETE final_report=$FINAL_AUDITED"
