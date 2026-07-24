# Runtime Setup

WorldArena separates **source code** (`thirdparty/`, git submodules) from **weights and caches** (`ckpt/` or `WORLDARENA_CHECKPOINT_ROOT`). This page covers environment setup, metric checkpoint downloads, third-party checkouts, and the evaluation workflow.

WorldArena is part of [OpenEnvision](https://openenvision.github.io/index.html), a joint research hub for open vision intelligence and world modeling.

| Section | What it covers |
|---------|----------------|
| [Setup Instructions](#setup-instructions) | Clone, `.env`, checkpoint root |
| [Evaluation Environment](#evaluation-environment-setup) | Conda `worldarena` env + metric runtime builds |
| [Download Checkpoints](#download-checkpoints-and-assets) | Metric weights, HF/Torch caches, one-click script |
| [Third-Party Repos](#third-party-repository-setup) | Submodule layout (`thirdparty/README.md`) |
| [Model Adapter Weights](#model-adapter-checkpoints) | Optional generation checkpoints |
| [Run Evaluation](#run-evaluation) | Manifest → generate → evaluate |
| [Physics Simulator](#physics-simulator-runtime) | External simulator track |

## Setup Instructions

#### 1. Clone the repository

```shell
git clone --recurse-submodules https://github.com/OpenEnvision/WorldArena.git WorldArena
cd WorldArena
```

If you already cloned without submodules:

```shell
git submodule update --init --recursive
```

See [`thirdparty/README.md`](https://github.com/OpenEnvision/WorldArena/blob/main/thirdparty/README.md) for the full submodule ↔ model config map.

#### 2. Configure environment paths

Create a `.env` file in the repository root before running evaluation or model adapters. The variables below are the ones WorldArena actually reads:

```sh
# Required only when ckpt/ lives outside the checkout
WORLDARENA_CHECKPOINT_ROOT=/path/to/ckpt

# Recommended for the tested CUDA 12.1 stack
CUDA_HOME=/path/to/cuda-12.1

# Optional: external physics simulator runtime
WORLDARENA_PHYSICS_PYTHON_BIN=/path/to/physics-pipeline-env/bin/python
```

`WORLDARENA_CHECKPOINT_ROOT` is the single checkpoint root. When unset, all checkpoint-like paths resolve under `WorldArena/ckpt`. When set, the same directory layout must be preserved (`huggingface/`, `torch/`, `official_metrics/`, model-specific folders, and so on).

Recommended cache exports (also set automatically when using `worldarena` CLI helpers):

```sh
export HF_HOME="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}/torch"
```

#### 3. Export the environment variables

After creating `.env`, export the variables in every shell that will run WorldArena:

!!! note "Repeat in every new terminal"
    This step must be repeated in every new terminal session.

```sh
set -a
source .env
set +a
export $(grep -v '^#' .env | xargs)
```

#### 4. API access (optional)

If you evaluate API-backed models from `config/models/`, put provider keys in a local `.secrets` file or export them in your shell. WorldArena does not commit API credentials.

## Evaluation Environment Setup

The open-source baseline is a single Conda environment named `worldarena`. Do not use the old `new` environment.

#### 1. Create and activate the environment

```shell
# Tested on CUDA 12.1
export CUDA_HOME=/path/to/cuda-12.1/

conda create -n worldarena python=3.10 -y
conda activate worldarena
```

#### 2. Install core dependencies

```shell
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -e .

conda install -c conda-forge decord=0.6.0 -y

python -m pip install \
  transformers==4.46.3 diffusers accelerate pyiqa open3d iopath pycocotools \
  pytorch-lightning torchmetrics timm==0.4.12 xformers==0.0.28.post3 \
  easydict roma jaxtyping lpips "pyglet<2" e3nn moviepy==1.0.3 \
  plyfile pillow-heif pycolmap imageio-ffmpeg OpenEXR kornia gsplat trimesh

python -m pip install \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html \
  pytorch3d

python -m pip install \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html \
  torch-scatter

python -m pip install causal-conv1d==1.5.0.post8 mamba-ssm==2.2.4

ln -sfn "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
  "$CONDA_PREFIX/bin/ffmpeg"
```

If conda-forge cannot solve `decord` in an already-populated environment, fall back to:

```shell
python -m pip install decord==0.6.0
```

#### 3. Install official metric runtime dependencies

Several headline metrics (`camera_error`, `reprojection_error`, `object_detection`, `motion_accuracy`, and related flow metrics) reuse upstream implementations through in-tree adapters. Initialize the metric-source submodules and build DROID-SLAM when you need the full official benchmark path:

WorldArena keeps only the metric adapters under
`worldarena/benchmark/official_backends/`. DROID-SLAM, SEA-RAFT,
GroundingDINO, SAM2, and VFIMamba are used directly from their `thirdparty/`
submodule checkouts.

```shell
mkdir -p thirdparty

git submodule update --init --recursive thirdparty/DROID-SLAM
cd thirdparty/DROID-SLAM/
python setup.py install
cd ../..

git submodule update --init thirdparty/GroundingDINO
python -m pip install --no-build-isolation -e thirdparty/GroundingDINO

git submodule update --init thirdparty/sam2
cd thirdparty/sam2/
python -m pip install -e .
cd ../..

git submodule update --init thirdparty/SEA-RAFT thirdparty/VFIMamba
```

Install additional native metric runtimes (submodules where registered, otherwise clone once):

```shell
git submodule update --init \
  thirdparty/AMT \
  thirdparty/RAFT \
  thirdparty/Pulse-of-Motion \
  thirdparty/Depth-Anything-3 \
  thirdparty/met3r

# Not registered as submodules yet — clone manually when needed
git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything thirdparty/Video-Depth-Anything
git clone --depth 1 https://github.com/soCzech/TransNetV2 thirdparty/TransNetV2
git -C thirdparty/TransNetV2 lfs pull
```

Optional suite-level distribution metrics (separate upstream env pins):

```shell
git clone --depth 1 https://github.com/MCG-NKU/FVMD-frechet-video-motion-distance thirdparty/FVMD-frechet-video-motion-distance
git clone --depth 1 https://github.com/MCG-NKU/JEDi thirdparty/JEDi
```

For `hpsv3_norm`, install the inferencer package in the same `worldarena` env:

```shell
python -m pip install hpsv3
```

Optional MegaSAM backend for `camera_error` (kept in a separate conda environment):

```shell
# Set MEGASAM_PYTHON to the MegaSAM env after installing upstream mega-sam
export MEGASAM_PYTHON=/path/to/mega_sam/bin/python
```

#### 4. Verify the environment

```shell
python -m pip check
python -m pytest \
  tests/test_checkpoints_root.py \
  tests/test_long_sequence_runtime.py \
  tests/test_long_sequence_backend_runtime.py \
  tests/test_flow_sources.py \
  tests/test_depth_collision_metric.py \
  tests/test_reconstruction_consistency_metric.py \
  tests/test_quality_backends.py \
  tests/test_wonderworld_adapter.py \
  tests/test_flashworld_adapter.py
```

The environment prepared for the current evaluation run uses `decord==0.6.0`, `pytorch3d==0.7.8`, `pyiqa==0.1.15.post2`, `open3d==0.19.0`, `xformers==0.0.28.post3`, `torch-scatter==2.1.2+pt25cu121`, `causal-conv1d==1.5.0.post8`, and `mamba-ssm==2.2.4`.

If `pip check` reports only `decord 0.6.0 is not supported on this platform` after installing from PyPI, the package is usually importable but its published wheel metadata is tagged for CPython 3.6. Prefer conda-forge `decord`; otherwise treat that single `decord` metadata warning separately and verify with `python -c "import decord; print(decord.__version__)"`.

## Download Checkpoints and Assets

All commands below assume you are in the WorldArena checkout and that `WORLDARENA_CHECKPOINT_ROOT` points at the active checkpoint root when you use a shared disk layout.

!!! tip "One-click download"
    ```shell
    bash scripts/download_metric_checkpoints.sh          # core metric + official runtime weights
    bash scripts/download_metric_checkpoints.sh --all    # + HPSv3, JEDi, TransNetV2, HF/Torch caches
    ```

Missing checkpoints raise `FileNotFoundError` during evaluation — WorldArena does not download weights online at score time.

### Metric ↔ checkpoint quick reference

| Metric(s) | Path under `ckpt/` | Download source |
|-----------|-------------------|-----------------|
| `depth_collision` | `Video-Depth-Anything/metric_video_depth_anything_vits.pth` | [HF: depth-anything/Metric-Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small) |
| `long_sequence_*`, `long_horizon_diagnostics` | `amt_model/amt-s.pth`, `raft_model/models/raft-things.pth`, `musiq/musiq_spaq_ckpt-358bb6af.pth` | [HF: lalala125/AMT](https://huggingface.co/lalala125/AMT), [RAFT Dropbox models.zip](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip), [IQA-PyTorch release](https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth) |
| `temporal_calibration`, `physical_fps_*` | `Pulse-of-Motion/vc_common_10_60fps.ckpt` | [HF: xiangbog/Visual_Chronometer](https://huggingface.co/xiangbog/Visual_Chronometer) |
| `camera_error`, `reprojection_error` (DROID) | `official_metrics/droid.pth` | [Google Drive via gdown](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh) |
| `object_detection`, `motion_accuracy` | `official_metrics/groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth`, `sam2.1_hiera_base_plus.pt`, `bert-base-uncased/` | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth), [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), [HF: facebook/sam2.1-hiera-base-plus](https://huggingface.co/facebook/sam2.1-hiera-base-plus), [HF: google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| `optical_flow_aepe`, `motion_magnitude`, `motion_accuracy` | `official_metrics/Tartan-C-T-TSKH-spring540x960-M.pth`, `official_metrics/models/raft-things.pth` | [HF: Yarimasune/SEA-RAFT](https://huggingface.co/Yarimasune/SEA-RAFT), copy from `raft_model/` |
| `motion_smoothness` | `official_metrics/VFIMamba.pkl` | [HF: MCG-NJU/VFIMamba](https://huggingface.co/MCG-NJU/VFIMamba) (`model.pkl` copied as `VFIMamba.pkl`) |
| `long_sequence_aesthetic_quality` | `official_metrics/sac+logos+ava1-l14-linearMSE.pth` | [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth) |
| `depth_accuracy` | `huggingface/hub` → `depth-anything/Depth-Anything-V2-Small-hf` | [HF model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| `reconstruction_consistency`, `met3r_consistency` | `DA3-GIANT/`, HF `naver/MASt3R_*`, `torch/hub` DINO | [HF: depth-anything/DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT), [HF: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric) |
| `semantic_alignment`, `background_consistency`, `memory_revisit_consistency` | HF `openai/clip-vit-base-patch16`, `facebook/dinov2-base` | [CLIP](https://huggingface.co/openai/clip-vit-base-patch16), [DINOv2](https://huggingface.co/facebook/dinov2-base) |
| `hpsv3_norm` | `HPSv3/HPSv3.safetensors`, `HPSv3/HPSv3_7B_local.yaml`, `Qwen2-VL-7B-Instruct/` | [HF: MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3), [config template](https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml), [HF: Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| `segment_continuity` | `TransNetV2/transnetv2-pytorch-weights.pth` + `thirdparty/TransNetV2/inference-pytorch/` | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) |
| `jedi` (video suite) | `JEDi/vith16.pth.tar`, `JEDi/ssv2-probe.pth.tar` | [Meta JEPA weights](https://dl.fbaipublicfiles.com/jepa/vith16/) |

The in-tree adapters resolve files directly from `official_metrics/` at evaluation time. Keep filenames exactly as in the table above.

#### 1. Benchmark metric weights

```shell
CKPT="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}"

mkdir -p "$CKPT"/{Video-Depth-Anything,amt_model,raft_model/models,Pulse-of-Motion,musiq,official_metrics/models}

wget -q -O "$CKPT/Video-Depth-Anything/metric_video_depth_anything_vits.pth" \
  https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth

wget -q -O "$CKPT/amt_model/amt-s.pth" \
  https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth

wget -q -P /tmp https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip -o /tmp/models.zip -d "$CKPT/raft_model"

python -m pip install -q huggingface_hub
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
root = os.environ.get("WORLDARENA_CHECKPOINT_ROOT", "ckpt")
path = hf_hub_download(
    repo_id="xiangbog/Visual_Chronometer",
    filename="vc_common_10_60fps.ckpt",
    local_dir=f"{root}/Pulse-of-Motion",
)
print(path)
PY

wget -q -O "$CKPT/musiq/musiq_spaq_ckpt-358bb6af.pth" \
  https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
```

#### 2. Official metric runtime weights

These weights back the official camera and flow metrics (`object_detection`, `motion_accuracy`, camera/flow metrics):

```shell
CKPT="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}"
export CKPT

wget -q -P "$CKPT/official_metrics" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget -q -P "$CKPT/official_metrics" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget -q -P "$CKPT/official_metrics" \
  https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt

cp "$CKPT/raft_model/models/raft-things.pth" "$CKPT/official_metrics/models/raft-things.pth"

python -m pip install -q gdown
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O "$CKPT/official_metrics/droid.pth"

python - <<'PY'
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
  python - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir=os.path.join(os.environ["CKPT"], "official_metrics", "bert-base-uncased"),
)
PY
fi
```

#### 3. Quality / distribution metric extras

Required when running `hpsv3_norm`, `segment_continuity`, or the video-suite `jedi` metric:

```shell
CKPT="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}"

# HPSv3 reward model + Qwen2-VL backbone
wget -q -O "$CKPT/HPSv3/HPSv3.safetensors" \
  https://huggingface.co/MizzenAI/HPSv3/resolve/main/HPSv3.safetensors
wget -q -O "$CKPT/HPSv3/HPSv3_7B_local.yaml" \
  https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir "$CKPT/Qwen2-VL-7B-Instruct"

# JEDi / V-JEPA (video distribution metric)
wget -q -O "$CKPT/JEDi/vith16.pth.tar" \
  https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
wget -q -O "$CKPT/JEDi/ssv2-probe.pth.tar" \
  https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar

# TransNetV2 (segment_continuity): clone upstream, pull LFS weights, convert to PyTorch
git clone --depth 1 https://github.com/soCzech/TransNetV2 thirdparty/TransNetV2
git -C thirdparty/TransNetV2 lfs pull
cd thirdparty/TransNetV2/inference-pytorch && python convert_weights.py
cp transnetv2-pytorch-weights.pth "$CKPT/TransNetV2/"
```

Or run the bundled helper:

```shell
bash scripts/download_metric_checkpoints.sh --all
```

#### 4. Hugging Face and Torch caches

WorldArena pins HF/Torch caches under the checkpoint root and runs bundled adapters offline. Preload the backends you need before evaluation:

```shell
CKPT="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}"

export HF_HOME="$CKPT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CKPT/huggingface/hub"
export TRANSFORMERS_CACHE="$CKPT/transformers"
export TORCH_HOME="$CKPT/torch"

huggingface-cli download openai/clip-vit-base-patch16
huggingface-cli download facebook/dinov2-base
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf
huggingface-cli download depth-anything/DA3-GIANT --local-dir "$CKPT/DA3-GIANT"
huggingface-cli download naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

python - <<'PY'
import os
import torch
torch.hub.set_dir(os.environ.get("TORCH_HOME", "ckpt/torch"))
torch.hub.load("facebookresearch/dino:main", "dino_deitsmall16")
PY
```

Replace the Python one-liner with a small script if you prefer explicit shell exports; the important constraint is that DINO weights land under `ckpt/torch/hub/`.

## Model Adapter Checkpoints

Large model weights are optional. Add only the checkpoints for the adapters you plan to run from `config/models/`.

| Config | Upstream code | Weights under `ckpt/` |
|--------|---------------|----------------------|
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_video_dense.yaml` | `thirdparty/lingbot-video` | `lingbot-video-dense-1.3b` |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam` + Wan2.1 companion weights |
| `gen3c.yaml` | `thirdparty/GEN3C` | `nvidia--GEN3C-Cosmos-7B` |
| `matrix_game_35_first.yaml` | `thirdparty/Matrix-Game-3.5` | `Matrix-Game-3.5/*` |
| `matrix_game.yaml` | `thirdparty/Matrix-Game/Matrix-Game-3` | `Matrix-Game-3.0` |
| `matrix_game_1.yaml` | `thirdparty/Matrix-Game/Matrix-Game-1` | `Matrix-Game-1.0` |
| `alayaworld.yaml` | `thirdparty/AlayaWorld` | `AlayaWorld/*` + `gemma-3-12b-it-*` |
| `cami2v.yaml` | `thirdparty/CamI2V` | `CamI2V/512_cami2v_100k.pt` |
| `realcam_i2v.yaml` | `thirdparty/RealCam-I2V` | `RealCam-I2V/*.pt` + `CogVideoX1.5-5B-I2V` |
| `stable_virtual_camera.yaml` | `thirdparty/stable-virtual-camera` | `stable-virtual-camera/*` |
| `yume.yaml` | `thirdparty/YUME` | `Yume-5B-720P` |
| `abot_world.yaml` | `thirdparty/ABot-World` | `ABot-World-0-5B-LF` |

Example:

```shell
huggingface-cli download acvlab/ABot-World-0-5B-LF --local-dir ckpt/ABot-World-0-5B-LF
```

!!! tip "Offline evaluation contract"
    Missing checkpoints raise `FileNotFoundError` directly. WorldArena does not fall back to alternate cache roots or online downloads during evaluation.

## Third-Party Repository Setup

`thirdparty/` is for source checkouts and Git submodules only. Checkpoints, HF caches, and Torch caches stay under the active checkpoint root.

#### Directory contract

```text
WorldArena/
  ckpt/
    huggingface/
    torch/
    official_metrics/
    Video-Depth-Anything/
    amt_model/
    raft_model/
    Pulse-of-Motion/
    musiq/
    HPSv3/
    JEDi/
    TransNetV2/
    DA3-GIANT/
    Qwen2-VL-7B-Instruct/
    <model-specific-checkpoint-dirs>/
  thirdparty/          # git submodules — source only
    DROID-SLAM/
    GroundingDINO/
    sam2/
    SEA-RAFT/
    Video-Depth-Anything/
    TransNetV2/
    met3r/
    AMT/
    RAFT/
    Pulse-of-Motion/
    Depth-Anything-3/
    lingbot-world/
    ...
  artifacts/
    predictions/       # generated videos
    benchmark/reports/ # evaluation outputs
```

`artifacts/` and `.cache/` are disposable workspace output — not checkpoint storage.

#### Model adapter checkouts

Initialize only the submodules you need (names must match `config/models/*.yaml`):

```shell
git submodule update --init thirdparty/lingbot-world
git submodule update --init thirdparty/FlashWorld thirdparty/WonderWorld
```

See [`thirdparty/README.md`](https://github.com/OpenEnvision/WorldArena/blob/main/thirdparty/README.md) for the full list.

Practical rules:

- keep clone names aligned with bundled config defaults when possible
- pin upstream commits in local notes or experiment logs, not by committing source here
- accept upstream licenses separately for each repo and each weight release
- do not store checkpoints inside `thirdparty/`

Removed path mechanisms:

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend-specific fallback search chains

## Run Evaluation

All commands below assume `conda activate worldarena` and exported `.env` variables.

#### 1. Build the benchmark manifest

```shell
worldarena build-manifest --config config/benchmark.yaml
```

Use `worldarena build-inventory --config config/benchmark.yaml` only when you explicitly want the raw asset audit without immediately writing a manifest.

#### 2. Generate predictions or stage an existing folder

Generate with a bundled adapter:

```shell
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Or run the one-shot path:

```shell
worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Expected prediction layout:

```text
artifacts/predictions/<run-name>/
  image_static/<sample_id>.mp4
  image_dynamic/<sample_id>.mp4
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

#### 3. Evaluate predictions

Single-process evaluation:

```shell
worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root artifacts/predictions/lingbot_world
```

Useful filters:

- `--suites image_static,video_dynamic`
- `--limit 32`
- `--overwrite`
- `--model-config config/models/lingbot_world.yaml` when model-specific metric overrides should be applied before scoring

Evaluation artifacts are written under `artifacts/benchmark/reports/`:

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `leaderboard.csv`

Runs are directly comparable only when `missing_predictions = 0` and the same suites, manifest snapshot, and metric config were used.

!!! tip "Heaviest metric"
    `motion_accuracy` is the heaviest dynamic metric. The default benchmark config enables SAM2 video/state offload for 80 GB cards. On multi-GPU hosts, split devices through `benchmark.metric_runtime.motion_accuracy` (`sam2_device`, `flow_device`).

## Physics Simulator Runtime

The simulator-backed physics track is separate from the single `worldarena` metric environment above. WorldArena builds manifests, stages prediction videos, validates artifact contracts, reconstructs derived worldlines, and invokes an external simulator pipeline, but it does not vendor or install that pipeline.

Required external checkout contract:

```text
thirdparty/physics-pipeline/
  assets/<sample_id>.mp4
  assets/<sample_id>.json
  benchmark_export.py
  benchmark_eval.py
  main_part2.py
```

Runtime selection:

```shell
export WORLDARENA_PHYSICS_PYTHON_BIN=/path/to/physics-pipeline-env/bin/python
worldarena build-physics-manifest \
  --physics-pipeline-root thirdparty/physics-pipeline
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

If `WORLDARENA_PHYSICS_PYTHON_BIN` is unset, WorldArena falls back to the `python_bin` recorded in each physics case metadata file, then to the current Python executable.

Current repository-side validation for this path is:

```shell
python -m pytest tests/test_physics_simulator_derived_metrics.py
```

This verifies the in-tree derived simulator metrics and config registration. It does not prove that `benchmark_export.py`, `main_part2.py`, or `benchmark_eval.py` can run, because those belong to the external physics pipeline checkout.
