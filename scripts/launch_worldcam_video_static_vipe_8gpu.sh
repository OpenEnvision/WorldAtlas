#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena

PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/yangboxue/visual_generation/juanxi/conda/envs/worldcam/bin/python}"
MANIFEST="${MANIFEST:-artifacts/benchmark/manifest/worldarena_video_static_vipe_4971.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena/bench_output/worldcam-video-static-vipe}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" scripts/build_video_static_vipe_manifest.py --output "${MANIFEST}"
fi

"${PYTHON_BIN}" \
  scripts/launch_worldcam_persistent_shards.py \
  --model-config config/models/worldcam.yaml \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --suites video_static \
  --gpus "${GPUS}" \
  "$@"
