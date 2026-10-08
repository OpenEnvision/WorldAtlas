#!/usr/bin/env bash
# Download benchmark metric checkpoints into WORLDATLAS_CHECKPOINT_ROOT (default: ckpt/).
#
# Usage:
#   bash scripts/download_metric_checkpoints.sh            # core metric weights
#   bash scripts/download_metric_checkpoints.sh --hf-cache # also preload HF/Torch caches
#   bash scripts/download_metric_checkpoints.sh --all      # core + HF cache + distribution/dynamic extras
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${WORLDATLAS_CHECKPOINT_ROOT:-$ROOT/ckpt}"
export CKPT
DOWNLOAD_HF_CACHE=0
DOWNLOAD_ALL=0

usage() {
  cat <<EOF
Usage: bash scripts/download_metric_checkpoints.sh [--hf-cache] [--all]

  --hf-cache   Preload Hugging Face / Torch caches used by CLIP, DINOv2, depth, MEt3R, etc.
  --all        Download everything (core, FVMD, JEDi, TRAJAN, HPSv3, TransNet, HF cache)

Environment:
  WORLDATLAS_CHECKPOINT_ROOT   Checkpoint root (default: \$REPO/ckpt)
EOF
}

for arg in "$@"; do
  case "$arg" in
    --hf-cache) DOWNLOAD_HF_CACHE=1 ;;
    --all) DOWNLOAD_ALL=1; DOWNLOAD_HF_CACHE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$CKPT"/{Video-Depth-Anything,amt_model,raft_model/models,Pulse-of-Motion,musiq,official_metrics/models,HPSv3,JEDi,TRAJAN,TransNetV2,huggingface/hub,torch/hub/checkpoints}

echo "[1/6] Core bundled-metric weights (depth / flow / long-sequence / pulse / MUSIQ)..."

wget -q -O "$CKPT/Video-Depth-Anything/metric_video_depth_anything_vits.pth" \
  https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth

wget -q -O "$CKPT/amt_model/amt-s.pth" \
  https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth

if [[ ! -f "$CKPT/raft_model/models/raft-things.pth" ]]; then
  tmp_zip="$(mktemp /tmp/raft-models.XXXXXX.zip)"
  wget -q -O "$tmp_zip" https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
  unzip -o "$tmp_zip" -d "$CKPT/raft_model"
  rm -f "$tmp_zip"
fi

python -m pip install -q huggingface_hub
python - <<PY
import os
from huggingface_hub import hf_hub_download

root = os.environ["CKPT"]
path = hf_hub_download(
    repo_id="xiangbog/Visual_Chronometer",
    filename="vc_common_10_60fps.ckpt",
    local_dir=f"{root}/Pulse-of-Motion",
)
print(path)
PY

wget -q -O "$CKPT/musiq/musiq_spaq_ckpt-358bb6af.pth" \
  https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth

echo "[2/6] Official metric runtime weights (DROID / GroundingDINO / SAM / SEA-RAFT / VFIMamba)..."

wget -q -O "$CKPT/official_metrics/groundingdino_swint_ogc.pth" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget -q -O "$CKPT/official_metrics/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget -q -O "$CKPT/official_metrics/sam2.1_hiera_base_plus.pt" \
  https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt

cp -f "$CKPT/raft_model/models/raft-things.pth" "$CKPT/official_metrics/models/raft-things.pth"

python -m pip install -q gdown
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O "$CKPT/official_metrics/droid.pth"

python - <<PY
import os
import shutil
from huggingface_hub import hf_hub_download

root = os.environ["CKPT"]
official = f"{root}/official_metrics"

tartan_src = hf_hub_download(
    repo_id="Yarimasune/SEA-RAFT",
    filename="Tartan-C-T-TSKH-spring540x960-M.pth",
)
shutil.copy2(tartan_src, f"{official}/Tartan-C-T-TSKH-spring540x960-M.pth")

vfimamba_src = hf_hub_download(repo_id="MCG-NJU/VFIMamba", filename="model.pkl")
shutil.copy2(vfimamba_src, f"{official}/VFIMamba.pkl")
print(tartan_src)
print(vfimamba_src)
PY

wget -q -O "$CKPT/official_metrics/sac+logos+ava1-l14-linearMSE.pth" \
  https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth

if command -v huggingface-cli >/dev/null 2>&1; then
  export HF_HOME="$CKPT/huggingface"
  export HUGGINGFACE_HUB_CACHE="$CKPT/huggingface/hub"
  huggingface-cli download google-bert/bert-base-uncased --local-dir "$CKPT/official_metrics/bert-base-uncased"
else
  python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir=os.path.join(os.environ["CKPT"], "official_metrics", "bert-base-uncased"),
)
PY
fi

if [[ "$DOWNLOAD_ALL" == "1" ]]; then
  echo "[3/6] HPSv3 + Qwen2-VL (hpsv3_norm)..."

  wget -q -O "$CKPT/HPSv3/HPSv3.safetensors" \
    https://huggingface.co/MizzenAI/HPSv3/resolve/main/HPSv3.safetensors
  python - <<'PY'
from pathlib import Path
import json, os, struct
path = Path(os.environ["CKPT"]) / "HPSv3" / "HPSv3.safetensors"
size = path.stat().st_size
with path.open("rb") as handle:
    header_len = struct.unpack("<Q", handle.read(8))[0]
    metadata = json.loads(handle.read(header_len))
max_end = max(int(value["data_offsets"][1]) for value in metadata.values() if "data_offsets" in value)
covered = 8 + header_len + max_end
if size < covered:
    raise SystemExit(f"HPSv3 checkpoint truncated: {path} has {size} bytes, metadata covers {covered}")
if size > covered:
    print(f"truncating extra {size - covered} trailing bytes from {path}")
    os.truncate(path, covered)
PY

  if [[ ! -f "$CKPT/HPSv3/HPSv3_7B_local.yaml" ]]; then
    wget -q -O "$CKPT/HPSv3/HPSv3_7B_local.yaml" \
      https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    export HF_HOME="$CKPT/huggingface"
    export HUGGINGFACE_HUB_CACHE="$CKPT/huggingface/hub"
    huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir "$CKPT/Qwen2-VL-7B-Instruct"
  else
    python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="Qwen/Qwen2-VL-7B-Instruct",
    local_dir=os.path.join(os.environ["CKPT"], "Qwen2-VL-7B-Instruct"),
)
PY
  fi

  echo "[4/6] FVMD, JEDi, and TRAJAN official metric runtimes..."
  if [[ ! -d "$ROOT/thirdparty/FVMD-frechet-video-motion-distance/fvmd" ]]; then
    git clone --depth 1 \
      https://github.com/DSL-Lab/FVMD-frechet-video-motion-distance \
      "$ROOT/thirdparty/FVMD-frechet-video-motion-distance"
  fi
  if [[ ! -d "$ROOT/thirdparty/JEDi/videojedi" ]]; then
    git clone --depth 1 https://github.com/oooolga/JEDi "$ROOT/thirdparty/JEDi"
  fi
  if [[ ! -d "$ROOT/thirdparty/tapnet/tapnet/trajan" ]]; then
    git clone --depth 1 https://github.com/google-deepmind/tapnet "$ROOT/thirdparty/tapnet"
  fi

  wget -q -O "$CKPT/torch/hub/checkpoints/pips2_weights.pth" \
    https://github.com/ljh0v0/FVMD-frechet-video-motion-distance/releases/download/pips2_weights/pips2_weights.pth
  wget -q -O "$CKPT/JEDi/vith16.pth.tar" \
    https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
  wget -q -O "$CKPT/JEDi/ssv2-probe.pth.tar" \
    https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
  wget -q -O "$CKPT/TRAJAN/bootstapir_checkpoint_v2.npy" \
    https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
  wget -q -O "$CKPT/TRAJAN/track_autoencoder_ckpt.npz" \
    https://storage.googleapis.com/dm-tapnet/trajan/track_autoencoder_ckpt.npz

  echo "[5/6] TransNetV2 PyTorch weights (segment_continuity)..."
  if [[ ! -f "$CKPT/TransNetV2/transnetv2-pytorch-weights.pth" ]]; then
    transnet_root="$ROOT/thirdparty/TransNetV2"
    if [[ ! -d "$transnet_root/inference/transnetv2-weights" ]]; then
      echo "  Cloning TransNetV2 into thirdparty/TransNetV2 (requires git-lfs)..."
      git clone --depth 1 https://github.com/soCzech/TransNetV2 "$transnet_root"
      if command -v git-lfs >/dev/null 2>&1; then
        git -C "$transnet_root" lfs pull
      fi
    fi
    if [[ -d "$transnet_root/inference-pytorch" ]]; then
      echo "  Converting TensorFlow weights to PyTorch (needs tensorflow==2.1)..."
      (cd "$transnet_root/inference-pytorch" && python convert_weights.py)
      cp -f "$transnet_root/inference-pytorch/transnetv2-pytorch-weights.pth" \
        "$CKPT/TransNetV2/transnetv2-pytorch-weights.pth"
    else
      echo "  Skipped TransNetV2 conversion; clone thirdparty/TransNetV2 and run inference-pytorch/convert_weights.py manually."
    fi
  fi
else
  echo "[3/6] Skipping HPSv3 / JEDi / TransNetV2 (pass --all to include)."
fi

if [[ "$DOWNLOAD_HF_CACHE" == "1" ]]; then
  echo "[6/6] Hugging Face / Torch caches..."
  export HF_HOME="$CKPT/huggingface"
  export HUGGINGFACE_HUB_CACHE="$CKPT/huggingface/hub"
  export TRANSFORMERS_CACHE="$CKPT/huggingface/hub"
  export TORCH_HOME="$CKPT/torch"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download openai/clip-vit-base-patch16
    huggingface-cli download facebook/dinov2-base
    huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf
    huggingface-cli download depth-anything/DA3-GIANT --local-dir "$CKPT/DA3-GIANT"
    huggingface-cli download naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
  else
    python - <<PY
from huggingface_hub import snapshot_download
import os

root = os.environ["CKPT"]
for repo in (
    "openai/clip-vit-base-patch16",
    "facebook/dinov2-base",
    "depth-anything/Depth-Anything-V2-Small-hf",
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
):
    snapshot_download(repo_id=repo)
snapshot_download(
    repo_id="depth-anything/DA3-GIANT",
    local_dir=f"{root}/DA3-GIANT",
)
PY
  fi

  python - <<PY
import os
import torch

torch.hub.set_dir(os.environ.get("TORCH_HOME", "ckpt/torch"))
torch.hub.load("facebookresearch/dino:main", "dino_deitsmall16")
print("DINO torch.hub cache ready")
PY
else
  echo "[6/6] Skipping HF/Torch cache preload (pass --hf-cache or --all)."
fi

echo "Done. Checkpoint root: $CKPT"
echo "See README.md#metric-evaluation-checkpoints and docs/en/thirdparty.md for the full metric ↔ path map."
