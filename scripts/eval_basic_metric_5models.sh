#!/usr/bin/env bash
# Evaluate basic-metric (image_static + image_dynamic) predictions for the 5 new
# models and write per-model reports. Reuses scripts/run_eval_shards.py (sharded
# `worldarena evaluate`) across the local GPUs.
#
# Prerequisite: metric weights (CLIP, DINOv2, depth, ...) must be present under
# ckpt/. Fetch them once with: bash scripts/download_metric_checkpoints.sh
#
# Usage:
#   bash scripts/eval_basic_metric_5models.sh
#   MODELS="matrix_game_35 lingbot_world_v2" bash scripts/eval_basic_metric_5models.sh
#   PREDICTIONS_SUFFIX=-smoke bash scripts/eval_basic_metric_5models.sh   # eval alt output dirs
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ARENA_ROOT="$(cd -- "$WORLD_ROOT/.." && pwd)"

MODELS="${MODELS:-abot_world alayaworld matrix_game_35 lingbot_world_v2 lingbot_video}"
SUITES="${SUITES:-image_static image_dynamic}"          # space-separated; run each separately
MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_manifest.jsonl}"
CONFIG="${CONFIG:-config/benchmark.yaml}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LIMIT="${LIMIT:-}"
MATRIX_GAME_35_PERSON="${MATRIX_GAME_35_PERSON:-first}"
LINGBOT_VIDEO_VARIANT="${LINGBOT_VIDEO_VARIANT:-dense}"

# Evaluation runtime python needs worldarena + metric deps.
DEFAULT_EVAL_PYTHON="$ARENA_ROOT/envs/alayaworld/bin/python"
if [[ ! -x "$DEFAULT_EVAL_PYTHON" ]]; then
    DEFAULT_EVAL_PYTHON="python"
fi
PYTHON_BIN="${EVAL_PYTHON_BIN:-$DEFAULT_EVAL_PYTHON}"
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
# NOTE: do NOT set WORLDARENA_CHECKPOINT_ROOT here. Evaluation reads metric
# weights (CLIP, DINOv2, depth, ...) from the canonical WorldArena/ckpt, which is
# where scripts/download_metric_checkpoints.sh installs them. Pointing the root at
# world_models (needed only for the generate stage) would hide the metric weights.

cd "$WORLD_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST; run scripts/prepare_basic_metric.sh first." >&2
    exit 2
fi

# model -> production output dir name (must match the launch scripts).
prediction_dir_for() {
    case "$1" in
        abot_world)       echo "abot-world-production" ;;
        alayaworld)       echo "alayaworld-production" ;;
        matrix_game_35)   echo "matrix-game-3.5-${MATRIX_GAME_35_PERSON}-production" ;;
        lingbot_world_v2) echo "lingbot-world-v2-production" ;;
        lingbot_video)    echo "lingbot-video-${LINGBOT_VIDEO_VARIANT}-production" ;;
        *) return 2 ;;
    esac
}

SUFFIX="${PREDICTIONS_SUFFIX:-}"
failed=()
for model in $MODELS; do
    pred_name="$(prediction_dir_for "$model")" || { echo "ERROR: unknown model '$model'" >&2; failed+=("$model"); continue; }
    pred_root="artifacts/predictions/${pred_name}${SUFFIX}"
    if [[ ! -d "$pred_root" ]]; then
        echo "[skip] $model: predictions dir missing ($pred_root)" >&2
        failed+=("$model")
        continue
    fi
    for suite in $SUITES; do
        out_dir="artifacts/benchmark/reports/${pred_name}${SUFFIX}-basic/${suite}"
        echo ""
        echo "==================== eval $model / $suite ===================="
        echo "predictions=$pred_root  ->  report=$out_dir"
        eval_args=(
            "$PYTHON_BIN" scripts/run_eval_shards.py
            --config "$CONFIG"
            --manifest "$MANIFEST"
            --predictions-root "$pred_root"
            --output-dir "$out_dir"
            --suites "$suite"
            --model-name "$pred_name"
            --gpus "$GPUS"
        )
        [[ -n "$LIMIT" ]] && eval_args+=(--limit "$LIMIT")
        if ! "${eval_args[@]}"; then
            echo "[FAILED] eval $model / $suite" >&2
            failed+=("$model/$suite")
        fi
    done
done

echo ""
echo "==================== summary ===================="
echo "Reports under: artifacts/benchmark/reports/*-basic/"
if (( ${#failed[@]} == 0 )); then
    echo "All evaluations completed for: $MODELS"
else
    echo "Failed / skipped: ${failed[*]}" >&2
    exit 1
fi
