#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena

MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_video_static_vipe_4971.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena/bench_output/matrix-game-3-video-static-vipe}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/matrix-game-3.0/bin/python \
  scripts/launch_matrix_game3_persistent_shards.py \
  --model-config config/models/matrix_game_3_video_static_vipe.yaml \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --suites video_static \
  --gpus "${GPUS}" \
  "$@"
