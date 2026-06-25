#!/usr/bin/env bash
# Monitor WonderWorld/WonderJourney image_static 8-GPU runs.
# Usage:
#   bash scripts/monitor_wonder_image_static_8gpu.sh <RUN_ID>
set -euo pipefail

WORLD_ROOT="/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena"
RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
    echo "usage: $0 <RUN_ID>" >&2
    exit 2
fi
LOG_BASE="$WORLD_ROOT/artifacts/logs"
OUTPUT_BASE="$WORLD_ROOT/artifacts/predictions"

status_file="$LOG_BASE/wonder_image_static_8gpu_${RUN_ID}.status"
echo "status_file=$status_file"
if [[ -f "$status_file" ]]; then
    tail -40 "$status_file"
else
    echo "status file not found"
fi

for model in wonderworld wonderjourney; do
    output_dir="$OUTPUT_BASE/${model}-image-static-8gpu-singleload-${RUN_ID}"
    log_dir="$LOG_BASE/${model}_image_static_8gpu_${RUN_ID}"
    echo
    echo "== $model =="
    echo "output_dir=$output_dir"
    if [[ -d "$output_dir" ]]; then
        echo "mp4_count=$(find "$output_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
        echo "jsonl_records=$(find "$output_dir" -maxdepth 1 -type f -name 'generation_records*.jsonl' -print -exec wc -l {} \; 2>/dev/null | tail -20)"
    else
        echo "output_dir not created yet"
    fi
    echo "log_dir=$log_dir"
    if [[ -f "$log_dir/shard_pids.txt" ]]; then
        echo "pids=$(tr '\n' ' ' < "$log_dir/shard_pids.txt")"
        while read -r pid; do
            [[ -z "$pid" ]] && continue
            ps -o pid,ppid,stat,etime,cmd -p "$pid" 2>/dev/null || true
        done < "$log_dir/shard_pids.txt"
    fi
    if [[ -d "$log_dir" ]]; then
        for log in "$log_dir"/shard*.log; do
            [[ -f "$log" ]] || continue
            shard="$(basename "$log")"
            echo "-- tail $shard --"
            tail -8 "$log" | tr '\r' '\n' | tail -8
        done
    fi
done
