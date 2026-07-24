#!/usr/bin/env bash
# Generic checkpoint-resident production launcher for every WorldArena model.
#
# One top-level worker is created per GPU group.  Each worker receives a whole
# manifest shard in one generate_batch call, so a verified adapter loads its
# checkpoint once and reuses the resident runtime for every sample in that
# shard. Idle GPU groups start independently when queueing is enabled. Models
# with nproc_per_node=8 automatically use one 8-GPU worker.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DEFAULT_PYTHON="$WORLD_ROOT/../conda/envs/worldarena/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${WORLDARENA_BATCH_PYTHON_BIN:-$DEFAULT_PYTHON}"

MODEL_CONFIG="${WORLDARENA_BATCH_MODEL_CONFIG:-${1:-}}"
if [[ -z "$MODEL_CONFIG" ]]; then
    echo "Usage: WORLDARENA_BATCH_MODEL_CONFIG=config/models/<model>.yaml $0" >&2
    echo "   or: $0 config/models/<model>.yaml" >&2
    exit 2
fi

BENCHMARK_CONFIG="${WORLDARENA_BATCH_BENCHMARK_CONFIG:-config/benchmark.yaml}"
DEFAULT_MANIFEST="artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl"
if [[ ! -f "$WORLD_ROOT/$DEFAULT_MANIFEST" ]]; then
    DEFAULT_MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"
fi
MANIFEST="${WORLDARENA_BATCH_MANIFEST:-$DEFAULT_MANIFEST}"
TOTAL_GPUS="${WORLDARENA_BATCH_TOTAL_GPUS:-8}"
GPU_OFFSET="${WORLDARENA_BATCH_GPU_OFFSET:-0}"
LIMIT="${WORLDARENA_BATCH_LIMIT:-}"
DETACH="${WORLDARENA_BATCH_DETACH:-1}"
OVERWRITE="${WORLDARENA_BATCH_OVERWRITE:-0}"
FAIL_FAST="${WORLDARENA_BATCH_FAIL_FAST:-0}"
DRY_RUN="${WORLDARENA_BATCH_DRY_RUN:-0}"
VALIDATE_PATHS="${WORLDARENA_BATCH_VALIDATE_PATHS:-1}"
WAIT_FOR_GPUS="${WORLDARENA_BATCH_WAIT_FOR_GPUS:-0}"
IDLE_MEMORY_MIB="${WORLDARENA_BATCH_IDLE_MEMORY_MIB:-1024}"
WAIT_POLL_SECONDS="${WORLDARENA_BATCH_GPU_WAIT_POLL_SECONDS:-30}"
IDLE_STABILITY_POLLS="${WORLDARENA_BATCH_GPU_STABILITY_POLLS:-3}"
MASTER_PORT_BASE="${WORLDARENA_BATCH_MASTER_PORT_BASE:-29640}"

cd "$WORLD_ROOT"
if [[ "$PYTHON_BIN" == */* ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "ERROR: Python executable does not exist: $PYTHON_BIN" >&2
        exit 2
    fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python command is unavailable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ "$PYTHON_BIN" == */* ]]; then
    export PATH="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd):$PATH"
fi
for required_path in "$BENCHMARK_CONFIG" "$MODEL_CONFIG" "$MANIFEST"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 2
    fi
done
if (( TOTAL_GPUS <= 0 || GPU_OFFSET < 0 || IDLE_STABILITY_POLLS <= 0 )); then
    echo "ERROR: TOTAL_GPUS and IDLE_STABILITY_POLLS must be positive; GPU_OFFSET must be non-negative" >&2
    exit 2
fi

preflight_args=(
    "$PYTHON_BIN" "$SCRIPT_DIR/model_batch_preflight.py"
    --model-config "$MODEL_CONFIG"
    --total-gpus "$TOTAL_GPUS"
    --require-load-once
    --format lines
)
if [[ "$VALIDATE_PATHS" == "0" || "$VALIDATE_PATHS" == "false" ]]; then
    preflight_args+=(--no-validate-paths)
fi
mapfile -t preflight < <("${preflight_args[@]}")
if (( ${#preflight[@]} != 7 )); then
    echo "ERROR: model preflight returned ${#preflight[@]} fields; expected 7" >&2
    exit 2
fi
MODEL_NAME="${preflight[0]}"
MODEL_SLUG="${preflight[1]}"
MODEL_FAMILY="${preflight[2]}"
CHECKPOINT_POLICY="${preflight[3]}"
GPUS_PER_WORKER="${preflight[4]}"
WORKER_COUNT="${preflight[5]}"
DEFAULT_SUITES="${preflight[6]}"
SUITES="${WORLDARENA_BATCH_SUITES:-$DEFAULT_SUITES}"

TIMESTAMP="${WORLDARENA_BATCH_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${WORLDARENA_BATCH_OUTPUT_DIR:-$WORLD_ROOT/artifacts/predictions/${MODEL_SLUG}-production}"
LOG_DIR="${WORLDARENA_BATCH_LOG_DIR:-$WORLD_ROOT/artifacts/logs/${MODEL_SLUG}_production_$TIMESTAMP}"
ACTIVE_PID_FILE="$OUTPUT_DIR/.worldarena_batch_active_pids"

if command -v nvidia-smi >/dev/null 2>&1; then
    PHYSICAL_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    if (( GPU_OFFSET + TOTAL_GPUS > PHYSICAL_GPU_COUNT )); then
        echo "ERROR: requested GPU range ends at $((GPU_OFFSET + TOTAL_GPUS - 1)), but only $PHYSICAL_GPU_COUNT GPUs exist" >&2
        exit 2
    fi
elif [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
    echo "ERROR: nvidia-smi is required for GPU production" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
if [[ -f "$ACTIVE_PID_FILE" && "${WORLDARENA_BATCH_ALLOW_CONCURRENT:-0}" != "1" ]]; then
    while IFS= read -r active_pid; do
        if [[ "$active_pid" =~ ^[0-9]+$ ]] && kill -0 "$active_pid" 2>/dev/null; then
            echo "ERROR: production worker pid=$active_pid is already active for $OUTPUT_DIR" >&2
            exit 3
        fi
    done < "$ACTIVE_PID_FILE"
fi

export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "[$(date --iso-8601=seconds)] checkpoint-resident production launcher"
echo "model=$MODEL_NAME family=$MODEL_FAMILY policy=$CHECKPOINT_POLICY"
echo "model_config=$MODEL_CONFIG manifest=$MANIFEST suites=$SUITES"
echo "gpus_per_worker=$GPUS_PER_WORKER workers=$WORKER_COUNT total_gpus=$TOTAL_GPUS gpu_offset=$GPU_OFFSET"
echo "output_dir=$OUTPUT_DIR log_dir=$LOG_DIR global_limit=${LIMIT:-<none>}"

gpu_group_is_idle() {
    local first_gpu="$1"
    local group_count="$2"
    local gpu_memory
    local gpu_index
    mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    for ((gpu_index=first_gpu; gpu_index<first_gpu+group_count; gpu_index++)); do
        if (( ${gpu_memory[$gpu_index]} > IDLE_MEMORY_MIB )); then
            return 1
        fi
    done
    return 0
}

gpu_range_is_idle() {
    gpu_group_is_idle "$GPU_OFFSET" "$TOTAL_GPUS"
}

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]] \
    && ! gpu_range_is_idle \
    && [[ "$WAIT_FOR_GPUS" != "1" && "$WAIT_FOR_GPUS" != "true" ]]; then
    echo "ERROR: requested GPUs are busy; set WORLDARENA_BATCH_WAIT_FOR_GPUS=1 to queue" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >&2
    exit 4
fi

pids=()
worker_pids=()
worker_states=()
worker_idle_counts=()
for ((worker_index=0; worker_index<WORKER_COUNT; worker_index++)); do
    worker_states[$worker_index]="pending"
    worker_idle_counts[$worker_index]=0
done

launched_count=0
while (( launched_count < WORKER_COUNT )); do
    waiting_workers=()
    for ((worker_index=0; worker_index<WORKER_COUNT; worker_index++)); do
        if [[ "${worker_states[$worker_index]}" != "pending" ]]; then
            continue
        fi
        first_gpu=$((GPU_OFFSET + worker_index * GPUS_PER_WORKER))
        last_gpu=$((first_gpu + GPUS_PER_WORKER - 1))
        if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]] \
            && ! gpu_group_is_idle "$first_gpu" "$GPUS_PER_WORKER"; then
            worker_idle_counts[$worker_index]=0
            if [[ "$WAIT_FOR_GPUS" == "1" || "$WAIT_FOR_GPUS" == "true" ]]; then
                waiting_workers+=("$worker_index:$first_gpu-$last_gpu")
                continue
            fi
            echo "ERROR: GPUs $first_gpu-$last_gpu became busy before worker=$worker_index launched" >&2
            exit 4
        fi
        if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]] \
            && [[ "$WAIT_FOR_GPUS" == "1" || "$WAIT_FOR_GPUS" == "true" ]]; then
            worker_idle_counts[$worker_index]=$((worker_idle_counts[$worker_index] + 1))
            if (( worker_idle_counts[$worker_index] < IDLE_STABILITY_POLLS )); then
                waiting_workers+=(
                    "$worker_index:$first_gpu-$last_gpu(${worker_idle_counts[$worker_index]}/$IDLE_STABILITY_POLLS)"
                )
                continue
            fi
        fi

        visible_gpus=()
        for ((group_index=0; group_index<GPUS_PER_WORKER; group_index++)); do
            visible_gpus+=("$((first_gpu + group_index))")
        done
        visible_csv="$(IFS=,; echo "${visible_gpus[*]}")"
        worker_log="$LOG_DIR/shard${worker_index}.log"
        worker_args=(
            "$PYTHON_BIN" -u -m worldarena.cli generate
            --config "$BENCHMARK_CONFIG"
            --model-config "$MODEL_CONFIG"
            --manifest "$MANIFEST"
            --suites "$SUITES"
            --output-dir "$OUTPUT_DIR"
            --num-shards "$WORKER_COUNT"
            --shard-index "$worker_index"
        )
        if [[ -n "$LIMIT" ]]; then
            worker_args+=(--limit "$LIMIT")
        fi
        if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
            worker_args+=(--overwrite)
        fi
        if [[ "$FAIL_FAST" == "1" || "$FAIL_FAST" == "true" ]]; then
            worker_args+=(--fail-fast)
        fi
        echo "[$(date --iso-8601=seconds)] worker=$worker_index visible_gpus=$visible_csv log=$worker_log"
        if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q WORLDARENA_LOCAL_MASTER_PORT=%q' "$visible_csv" "$((MASTER_PORT_BASE + worker_index))"
            printf ' %q' "${worker_args[@]}"
            printf ' >%q 2>&1\n' "$worker_log"
            worker_states[$worker_index]="dry-run"
        else
            CUDA_VISIBLE_DEVICES="$visible_csv" \
                WORLDARENA_LOCAL_MASTER_PORT="$((MASTER_PORT_BASE + worker_index))" \
                nohup "${worker_args[@]}" >"$worker_log" 2>&1 &
            worker_pid="$!"
            worker_pids[$worker_index]="$worker_pid"
            worker_states[$worker_index]="launched"
            pids+=("$worker_pid")
            printf '%s\n' "${pids[@]}" > "$ACTIVE_PID_FILE"
            printf '%s\n' "${pids[@]}" > "$LOG_DIR/shard_pids.txt"
        fi
        launched_count=$((launched_count + 1))
    done
    if (( launched_count < WORKER_COUNT )); then
        echo "[$(date --iso-8601=seconds)] waiting for stable-idle worker:GPU groups ${waiting_workers[*]} below ${IDLE_MEMORY_MIB} MiB"
        sleep "$WAIT_POLL_SECONDS"
    fi
done

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[$(date --iso-8601=seconds)] dry run complete; no workers started"
    exit 0
fi

printf '%s\n' "$OUTPUT_DIR" > "$LOG_DIR/output_dir.txt"
echo "[$(date --iso-8601=seconds)] launched pids=${pids[*]}"

if [[ "$DETACH" == "1" || "$DETACH" == "true" ]]; then
    echo "[$(date --iso-8601=seconds)] detached; logs=$LOG_DIR"
    exit 0
fi

failed=0
for ((worker_index=0; worker_index<WORKER_COUNT; worker_index++)); do
    if wait "${worker_pids[$worker_index]}"; then
        echo "[$(date --iso-8601=seconds)] worker $worker_index completed OK"
    else
        return_code=$?
        failed=1
        echo "[$(date --iso-8601=seconds)] worker $worker_index FAILED rc=$return_code" >&2
    fi
done
rm -f -- "$ACTIVE_PID_FILE"
echo "[$(date --iso-8601=seconds)] all workers finished; output_dir=$OUTPUT_DIR"
exit "$failed"
