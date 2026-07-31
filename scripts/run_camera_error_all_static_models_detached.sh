#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${WORLDARENA_PYTHON:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldarena/bin/python}"
CONFIG="${WORLDARENA_CONFIG:-config/benchmark.yaml}"
MANIFEST="${WORLDARENA_MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
PREDICTIONS_ROOT="${WORLDARENA_PREDICTIONS_ROOT:-artifacts/predictions}"
RUN_ID="${WORLDARENA_RUN_ID:-camera_error_all_image_static_20260726_v2}"
MAX_GAP_ATTEMPTS="${WORLDARENA_MAX_GAP_ATTEMPTS:-5}"
NUM_SHARDS="${WORLDARENA_NUM_SHARDS:-64}"
WAN_MODEL="${WORLDARENA_WAN_MODEL:-wan22-ti2v-5b-image-static-plan-missing}"
WAN_RESUME_OUTPUT="${WORLDARENA_WAN_RESUME_OUTPUT:-$ROOT/artifacts/benchmark/reports/wan22-ti2v-5b-image-static-camera-full-64shard-8gpu-20260725}"

REPORT_ROOT="$ROOT/artifacts/benchmark/reports"
RUN_ROOT="$REPORT_ROOT/$RUN_ID"
STATE_DIR="$ROOT/artifacts/benchmark/background_jobs/$RUN_ID"
MODELS_FILE="$STATE_DIR/models.txt"
STATUS_FILE="$STATE_DIR/status"
DONE_FILE="$STATE_DIR/DONE"
LOCK_FILE="$STATE_DIR/supervisor.lock"
PID_FILE="$STATE_DIR/supervisor.pid"
SUPERVISOR_LOG="$STATE_DIR/supervisor.log"

mkdir -p "$RUN_ROOT" "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another $RUN_ID supervisor already holds $LOCK_FILE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

detected_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
GPU_CSV="${WORLDARENA_GPUS:-$detected_gpus}"
if [[ -z "$GPU_CSV" ]]; then
  echo "No visible GPUs found" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
if ((${#GPU_IDS[@]} == 0)); then
  echo "No usable GPU IDs in WORLDARENA_GPUS=$GPU_CSV" >&2
  exit 2
fi

gpu_sequence=()
for ((index = 0; index < NUM_SHARDS; index += 1)); do
  gpu_sequence+=("${GPU_IDS[index % ${#GPU_IDS[@]}]}")
done
GPU_SEQUENCE="$(IFS=','; printf '%s' "${gpu_sequence[*]}")"
MAX_CONCURRENT_PER_GPU="${WORLDARENA_MAX_CONCURRENT_PER_GPU:-1}"
MAX_CONCURRENT="${WORLDARENA_MAX_CONCURRENT_SHARDS:-$(( ${#GPU_IDS[@]} * MAX_CONCURRENT_PER_GPU ))}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$SUPERVISOR_LOG"
}

write_status() {
  local stage="$1"
  local model="${2:-}"
  local detail="${3:-}"
  local temporary="$STATUS_FILE.tmp"
  {
    printf 'pid=%s\n' "$$"
    printf 'updated_at=%s\n' "$(timestamp)"
    printf 'stage=%s\n' "$stage"
    printf 'model=%s\n' "$model"
    printf 'detail=%s\n' "$detail"
    printf 'visible_gpus=%s\n' "$GPU_CSV"
    printf 'logical_shards=%s\n' "$NUM_SHARDS"
    printf 'run_root=%s\n' "$RUN_ROOT"
  } >"$temporary"
  mv "$temporary" "$STATUS_FILE"
}

SUCCESS=0
on_exit() {
  local exit_code=$?
  if (( SUCCESS == 0 )); then
    write_status "failed" "" "exit_code=$exit_code"
    log "Supervisor failed with exit code $exit_code"
  fi
}
trap on_exit EXIT

printf '%s\n' "$$" >"$PID_FILE"
rm -f "$DONE_FILE"

"$PYTHON_BIN" - "$MANIFEST" "$PREDICTIONS_ROOT" "$REPORT_ROOT" "$MODELS_FILE" "${WORLDARENA_MODELS:-}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
predictions_root = Path(sys.argv[2])
reports_root = Path(sys.argv[3])
output_path = Path(sys.argv[4])
requested = {item.strip() for item in sys.argv[5].split(",") if item.strip()}

manifest = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
static_rows = [row for row in manifest if row.get("suite") == "image_static"]
expected_ids = {str(row["sample_id"]) for row in static_rows}
expected_stems = {str(row["prediction_stem"]) for row in static_rows}


def has_complete_camera_error(scores_path: Path) -> bool:
    if not scores_path.is_file():
        return False
    valid_ids: set[str] = set()
    with scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != "image_static":
                continue
            metric = (row.get("metrics") or {}).get("camera_error") or {}
            if (
                metric.get("normalized") is not None
                and not metric.get("error")
                and metric.get("eligibility_status") != "not_applicable"
            ):
                valid_ids.add(str(row.get("sample_id")))
    return valid_ids == expected_ids


completed_roots: set[Path] = set()
for run_manifest_path in reports_root.rglob("run_manifest.json"):
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    prediction_root = run_manifest.get("predictions_root")
    if not prediction_root:
        continue
    metrics = set(run_manifest.get("metric_override") or [])
    reported = set((run_manifest.get("reported_metrics") or {}).get("image_static") or [])
    if "camera_error" not in metrics and "camera_error" not in reported:
        continue
    if has_complete_camera_error(run_manifest_path.parent / "per_sample_scores.jsonl"):
        completed_roots.add(Path(prediction_root).resolve())

candidates: list[str] = []
for root in sorted(predictions_root.iterdir()):
    if not root.is_dir() or root.resolve() in completed_roots:
        continue
    stems: set[str] = set()
    for pattern in ("*.mp4", "*.mov", "*.webm", "*.mkv"):
        stems.update(path.stem for path in root.glob(pattern))
    if expected_stems <= stems:
        candidates.append(root.name)

if requested:
    unknown = requested.difference(candidates)
    if unknown:
        raise SystemExit(f"requested model(s) are incomplete or already covered: {sorted(unknown)}")
    candidates = [name for name in candidates if name in requested]

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(candidates) + ("\n" if candidates else ""), encoding="utf-8")
print(json.dumps({"expected_samples": len(expected_ids), "completed": len(completed_roots), "queued": candidates}, ensure_ascii=False, indent=2))
PY

mapfile -t MODELS < "$MODELS_FILE"
PRIORITY_MODELS="${WORLDARENA_PRIORITY_MODELS:-wan22-ti2v-5b-image-static-plan-missing}"
IFS=',' read -r -a priority_names <<< "$PRIORITY_MODELS"
ordered_models=()
for priority_name in "${priority_names[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ "$model" == "$priority_name" ]]; then
      ordered_models+=("$model")
    fi
  done
done
for model in "${MODELS[@]}"; do
  already_ordered=0
  for ordered_model in "${ordered_models[@]}"; do
    if [[ "$model" == "$ordered_model" ]]; then
      already_ordered=1
      break
    fi
  done
  if (( already_ordered == 0 )); then
    ordered_models+=("$model")
  fi
done
MODELS=("${ordered_models[@]}")
if [[ "${WORLDARENA_DRY_RUN:-0}" == "1" ]]; then
  SUCCESS=1
  write_status "dry_run" "" "queued_models=${#MODELS[@]}"
  log "Dry run complete: queued_models=${#MODELS[@]}"
  exit 0
fi
if ((${#MODELS[@]} == 0)); then
  SUCCESS=1
  write_status "complete" "" "no pending complete image_static prediction roots"
  printf 'completed_at=%s\nqueued_models=0\n' "$(timestamp)" >"$DONE_FILE"
  log "No pending complete image_static prediction roots"
  exit 0
fi

run_sharded() {
  local model="$1"
  local manifest_path="$2"
  local output_dir="$3"
  local phase="$4"
  local exit_code

  write_status "$phase" "$model" "manifest=$manifest_path output=$output_dir"
  log "START model=$model phase=$phase output=$output_dir"
  set +e
  "$PYTHON_BIN" scripts/run_eval_shards.py \
    --config "$CONFIG" \
    --manifest "$manifest_path" \
    --predictions-root "$PREDICTIONS_ROOT/$model" \
    --output-dir "$output_dir" \
    --suites image_static \
    --metrics camera_error \
    --model-name "$model" \
    --gpus "$GPU_SEQUENCE" \
    --max-concurrent-shards "$MAX_CONCURRENT" \
    --max-concurrent-per-gpu "$MAX_CONCURRENT_PER_GPU" \
    --heartbeat-interval 30
  exit_code=$?
  set -e
  log "END model=$model phase=$phase exit_code=$exit_code"
  return "$exit_code"
}

gap_count() {
  awk 'END { print NR + 0 }' "$1"
}

complete_model() {
  local model="$1"
  local model_dir="$RUN_ROOT/$model"
  local full_output="$model_dir/full"
  local audited_output="$model_dir/audited"
  local gap_manifest
  local retry_output
  local attempt
  local count
  local -a sources

  mkdir -p "$model_dir"
  if [[ "$model" == "$WAN_MODEL" && -d "$WAN_RESUME_OUTPUT" ]]; then
    full_output="$WAN_RESUME_OUTPUT"
    log "RESUME model=$model existing_output=$full_output"
  fi
  run_sharded "$model" "$MANIFEST" "$full_output" "full" || true
  if [[ ! -s "$full_output/per_sample_scores.jsonl" ]]; then
    log "INCOMPLETE model=$model reason=no full score report"
    return 1
  fi

  sources=("$full_output")
  for ((attempt = 1; attempt <= MAX_GAP_ATTEMPTS; attempt += 1)); do
    gap_manifest="$model_dir/gap_r${attempt}.jsonl"
    local -a gap_args=(
      scripts/build_metric_gap_manifest.py
      --manifest "$MANIFEST"
      --suite image_static
      --metrics camera_error
      --output "$gap_manifest"
    )
    local source
    for source in "${sources[@]}"; do
      gap_args+=(--report "$source")
    done
    "$PYTHON_BIN" "${gap_args[@]}" | tee -a "$SUPERVISOR_LOG"
    count="$(gap_count "$gap_manifest")"
    log "GAP model=$model attempt=$attempt count=$count"
    if (( count == 0 )); then
      break
    fi

    retry_output="$model_dir/gap-r${attempt}"
    run_sharded "$model" "$gap_manifest" "$retry_output" "gap_r${attempt}" || true
    if [[ -s "$retry_output/per_sample_scores.jsonl" ]]; then
      sources+=("$retry_output")
    else
      log "GAP model=$model attempt=$attempt produced no score report"
    fi
  done

  gap_manifest="$model_dir/gap_final.jsonl"
  local -a final_gap_args=(
    scripts/build_metric_gap_manifest.py
    --manifest "$MANIFEST"
    --suite image_static
    --metrics camera_error
    --output "$gap_manifest"
  )
  local source
  for source in "${sources[@]}"; do
    final_gap_args+=(--report "$source")
  done
  "$PYTHON_BIN" "${final_gap_args[@]}" | tee -a "$SUPERVISOR_LOG"
  count="$(gap_count "$gap_manifest")"
  if (( count != 0 )); then
    log "INCOMPLETE model=$model remaining_gap=$count"
    return 1
  fi

  write_status "merge" "$model" "output=$audited_output"
  local -a merge_args=(
    scripts/merge_metric_reports.py
    --manifest "$MANIFEST"
    --suite image_static
    --metrics camera_error
    --predictions-root "$PREDICTIONS_ROOT/$model"
    --model-name "$model"
    --output-dir "$audited_output"
    --expected 3372
  )
  for source in "${sources[@]}"; do
    merge_args+=(--report "$source")
  done
  "$PYTHON_BIN" "${merge_args[@]}" | tee -a "$SUPERVISOR_LOG"
  log "COMPLETE model=$model output=$audited_output"
}

log "Supervisor pid=$$ queued_models=${#MODELS[@]} visible_gpus=$GPU_CSV logical_shards=$NUM_SHARDS max_concurrent=$MAX_CONCURRENT"
failures=0
for model in "${MODELS[@]}"; do
  if ! complete_model "$model"; then
    failures=$((failures + 1))
  fi
done

if (( failures != 0 )); then
  write_status "incomplete" "" "failed_models=$failures queued_models=${#MODELS[@]}"
  log "Queue incomplete: failed_models=$failures"
  exit 1
fi

SUCCESS=1
write_status "complete" "" "queued_models=${#MODELS[@]}"
printf 'completed_at=%s\nqueued_models=%s\nrun_root=%s\n' "$(timestamp)" "${#MODELS[@]}" "$RUN_ROOT" >"$DONE_FILE"
log "QUEUE COMPLETE queued_models=${#MODELS[@]}"
