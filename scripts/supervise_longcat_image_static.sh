#!/usr/bin/env bash
# Supervisor for the LongCat-Video image_static 8-GPU run.
#
# - Adopts the already-running 8 shards (shard i on GPU i) for OUTPUT_DIR.
# - Every CHECK_INTERVAL seconds logs a status snapshot (alive shards, mp4 count, GPU mem).
# - If a shard's orchestrator dies BEFORE writing generation_manifest.shard<i>.json,
#   it is treated as a crash and relaunched on the same GPU / shard-index. The runner
#   skips already-generated mp4s (no --overwrite) and loads the ckpt exactly once, so
#   the "one resident process per GPU, one model load" invariant is preserved.
# - Exits when every shard has produced its generation_manifest.shard<i>.json and no
#   shard orchestrator remains alive.
set -uo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
PYTHON_BIN="/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldfm/bin/python"
CHECKPOINT_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt"
MODEL_CONFIG="config/models/longcat_video.yaml"
BENCHMARK_CONFIG="config/benchmark_openenvision.yaml"
MANIFEST="artifacts/benchmark/manifest/worldarena_manifest.jsonl"

NUM_SHARDS="${NUM_SHARDS:-8}"
OUTPUT_DIR="${LONGCAT_OUTPUT_DIR:?set LONGCAT_OUTPUT_DIR}"
LOG_DIR="${LONGCAT_LOG_DIR:?set LONGCAT_LOG_DIR}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
MONITOR_LOG="$LOG_DIR/supervisor.log"

cd "$WORLD_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export WORLDARENA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date --iso-8601=seconds)] $*" >> "$MONITOR_LOG"; }

shard_alive() {  # $1 = shard index -> echoes pid if an orchestrator for this shard is running
    pgrep -af "worldarena.cli generate" \
        | grep -F -- "--output-dir $OUTPUT_DIR " \
        | grep -E -- "--shard-index $1\$" \
        | awk '{print $1}' | head -1
}

shard_done() {  # $1 = shard index -> returns 0 if completed manifest exists
    [[ -f "$OUTPUT_DIR/generation_manifest.shard$1.json" ]]
}

launch_shard() {  # $1 = shard index
    local i="$1"
    local shard_log="$LOG_DIR/shard${i}.log"
    log "RESTART shard $i on GPU $i -> $shard_log"
    CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON_BIN" -u -m worldarena.cli generate \
        --config "$BENCHMARK_CONFIG" \
        --model-config "$MODEL_CONFIG" \
        --manifest "$MANIFEST" \
        --suites image_static \
        --output-dir "$OUTPUT_DIR" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" \
        >> "$shard_log" 2>&1 &
    echo "$!" > "$LOG_DIR/restart_shard${i}.pid"
}

log "supervisor started: output_dir=$OUTPUT_DIR num_shards=$NUM_SHARDS interval=${CHECK_INTERVAL}s"

declare -A just_restarted
while true; do
    alive_count=0
    done_count=0
    status_line=""
    for ((i=0; i<NUM_SHARDS; i++)); do
        pid="$(shard_alive "$i")"
        if [[ -n "$pid" ]]; then
            alive_count=$((alive_count+1))
            status_line+=" s$i=alive($pid)"
            just_restarted[$i]=0
        elif shard_done "$i"; then
            done_count=$((done_count+1))
            status_line+=" s$i=DONE"
        else
            # not alive, not done -> crashed; relaunch unless we just did
            if [[ "${just_restarted[$i]:-0}" == "1" ]]; then
                status_line+=" s$i=restarting"
            else
                status_line+=" s$i=CRASHED"
                launch_shard "$i"
                just_restarted[$i]=1
            fi
        fi
    done

    mp4_count=$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' )
    log "alive=$alive_count done=$done_count mp4=$mp4_count gpu_mem_used=[$gpu_mem]${status_line}"

    if [[ "$done_count" -eq "$NUM_SHARDS" && "$alive_count" -eq 0 ]]; then
        log "ALL $NUM_SHARDS shards completed. Supervisor exiting."
        break
    fi
    sleep "$CHECK_INTERVAL"
done
