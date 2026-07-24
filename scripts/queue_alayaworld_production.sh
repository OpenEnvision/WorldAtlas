#!/usr/bin/env bash
# Queue the full AlayaWorld run behind a one-GPU, 45-chunk persistent smoke.
# Formal 8-GPU production starts only after the smoke video passes decoding metadata checks.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${ALAYAWORLD_PYTHON_BIN:-$WORLD_ROOT/../conda/envs/alayaworld/bin/python}"
REPO_ROOT="${ALAYAWORLD_REPO_ROOT:-$WORLD_ROOT/../github_repos/AlayaWorld}"
CHECKPOINT_DIR="${ALAYAWORLD_CHECKPOINT_DIR:-$WORLD_ROOT/../ckpt/AlayaWorld}"
SMOKE_SPEC="${ALAYAWORLD_SMOKE_BATCH_SPEC:-$WORLD_ROOT/outputs/alayaworld_integration/persistent_long_smoke.jsonl}"
SMOKE_GENERATION="${ALAYAWORLD_SMOKE_GENERATION_CONFIG:-$WORLD_ROOT/artifacts/predictions/alayaworld-production/_batch_specs/alayaworld_dd701ea634d1402e97882b50b9f71b35.generation.json}"
SMOKE_OUTPUT="${ALAYAWORLD_SMOKE_OUTPUT:-$WORLD_ROOT/outputs/alayaworld_integration/persistent_push_in_45chunk.mp4}"
QUEUE_LOG_DIR="${ALAYAWORLD_QUEUE_LOG_DIR:-$WORLD_ROOT/artifacts/logs/alayaworld_queue_$(date -u +%Y%m%d_%H%M%S)}"
IDLE_MEMORY_MIB="${ALAYAWORLD_QUEUE_IDLE_MEMORY_MIB:-1024}"
POLL_SECONDS="${ALAYAWORLD_QUEUE_POLL_SECONDS:-30}"
STABILITY_POLLS="${ALAYAWORLD_QUEUE_STABILITY_POLLS:-3}"
MAX_SMOKE_ATTEMPTS="${ALAYAWORLD_QUEUE_MAX_SMOKE_ATTEMPTS:-10}"
NO_EVAL_STABILITY_POLLS="${ALAYAWORLD_COMPETING_JOB_STABILITY_POLLS:-6}"

cd "$WORLD_ROOT"
for required_path in "$PYTHON_BIN" "$REPO_ROOT" "$CHECKPOINT_DIR" "$SMOKE_SPEC" "$SMOKE_GENERATION"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required AlayaWorld queue path is missing: $required_path" >&2
        exit 2
    fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required" >&2
    exit 2
fi
mkdir -p "$QUEUE_LOG_DIR" "$(dirname -- "$SMOKE_OUTPUT")"
printf '%s\n' "$$" > "$QUEUE_LOG_DIR/queue_pid.txt"

export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

idle_gpu_once() {
    local gpu_memory
    local gpu_index
    mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    for ((gpu_index=0; gpu_index<${#gpu_memory[@]}; gpu_index++)); do
        if (( ${gpu_memory[$gpu_index]} <= IDLE_MEMORY_MIB )); then
            printf '%s\n' "$gpu_index"
            return 0
        fi
    done
    return 1
}

wait_for_stable_idle_gpu() {
    local candidate=""
    local stable_count=0
    local observed=""
    while true; do
        observed="$(idle_gpu_once || true)"
        if [[ -n "$observed" && "$observed" == "$candidate" ]]; then
            stable_count=$((stable_count + 1))
        elif [[ -n "$observed" ]]; then
            candidate="$observed"
            stable_count=1
        else
            candidate=""
            stable_count=0
        fi
        if (( stable_count >= STABILITY_POLLS )); then
            printf '%s\n' "$candidate"
            return 0
        fi
        echo "[$(date --iso-8601=seconds)] waiting for one stable idle GPU (threshold=${IDLE_MEMORY_MIB}MiB)" >&2
        sleep "$POLL_SECONDS"
    done
}

valid_smoke_output() {
    if [[ ! -s "$SMOKE_OUTPUT" ]]; then
        return 1
    fi
    "$PYTHON_BIN" - "$SMOKE_OUTPUT" <<'PY'
import sys
import cv2

capture = cv2.VideoCapture(sys.argv[1])
if not capture.isOpened():
    raise SystemExit(1)
frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
capture.release()
duration = frames / fps if fps > 0.0 else 0.0
if frames < 1400 or duration < 55.0:
    raise SystemExit(1)
PY
}

competing_eval_jobs_active() {
    ps -eo pid=,args= | awk -v self_pid="$$" '
        $1 != self_pid && $0 ~ /[ \/]scripts\/run_eval_shards\.py([[:space:]]|$)/ {
            found = 1
        }
        END { exit(found ? 0 : 1) }
    '
}

wait_for_no_competing_eval_jobs() {
    local stable_count=0
    while (( stable_count < NO_EVAL_STABILITY_POLLS )); do
        if competing_eval_jobs_active; then
            stable_count=0
            echo "[$(date --iso-8601=seconds)] waiting for competing run_eval_shards.py jobs to finish"
        else
            stable_count=$((stable_count + 1))
            echo "[$(date --iso-8601=seconds)] no competing eval jobs ($stable_count/$NO_EVAL_STABILITY_POLLS stable polls)"
        fi
        if (( stable_count < NO_EVAL_STABILITY_POLLS )); then
            sleep "$POLL_SECONDS"
        fi
    done
}

echo "[$(date --iso-8601=seconds)] queued AlayaWorld long smoke then production"
echo "queue_log_dir=$QUEUE_LOG_DIR"

if ! valid_smoke_output; then
    smoke_ok=0
    for ((attempt=1; attempt<=MAX_SMOKE_ATTEMPTS; attempt++)); do
        smoke_gpu="$(wait_for_stable_idle_gpu)"
        attempt_log="$QUEUE_LOG_DIR/long_smoke_attempt${attempt}_gpu${smoke_gpu}.log"
        echo "[$(date --iso-8601=seconds)] long smoke attempt=$attempt gpu=$smoke_gpu log=$attempt_log"
        if CUDA_VISIBLE_DEVICES="$smoke_gpu" "$PYTHON_BIN" -u \
            -m worldarena.models.adapters.alayaworld_batch_runner \
            --repo_root "$REPO_ROOT" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --batch_spec_path "$SMOKE_SPEC" \
            --generation_config_path "$SMOKE_GENERATION" \
            >"$attempt_log" 2>&1; then
            if valid_smoke_output; then
                smoke_ok=1
                echo "[$(date --iso-8601=seconds)] long smoke validated"
                break
            fi
            echo "[$(date --iso-8601=seconds)] smoke command completed but video validation failed" >&2
        else
            return_code=$?
            echo "[$(date --iso-8601=seconds)] smoke attempt failed rc=$return_code" >&2
        fi
    done
    if (( smoke_ok != 1 )); then
        echo "ERROR: AlayaWorld long smoke failed after $MAX_SMOKE_ATTEMPTS attempts" >&2
        exit 5
    fi
else
    echo "[$(date --iso-8601=seconds)] existing long smoke already passes validation"
fi

wait_for_no_competing_eval_jobs
echo "[$(date --iso-8601=seconds)] smoke gate passed; starting each production shard as its GPU becomes idle"
export ALAYAWORLD_WAIT_FOR_GPUS=1
# Keep the launcher alive as the supervisor. Detached children are reclaimed by
# some managed DSW runtimes as soon as their launching session exits.
export ALAYAWORLD_DETACH=0
export ALAYAWORLD_NUM_SHARDS=8
export WORLDARENA_BATCH_GPU_STABILITY_POLLS="${ALAYAWORLD_PRODUCTION_STABILITY_POLLS:-6}"
export ALAYAWORLD_FAIL_FAST=1
export ALAYAWORLD_LIMIT="${ALAYAWORLD_LIMIT:-}"
exec "$SCRIPT_DIR/launch_alayaworld_8gpu.sh"
