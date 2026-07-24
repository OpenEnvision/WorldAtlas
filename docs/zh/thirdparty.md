# 运行时布局

WorldArena 将 **源码**（`thirdparty/` git 子模块）与 **权重和缓存**（`ckpt/` 或 `WORLDARENA_CHECKPOINT_ROOT`）分开管理。本页涵盖环境配置、metric checkpoint 下载、第三方 checkout 与 evaluation 工作流。

WorldArena 是 [OpenEnvision](https://openenvision.github.io/index.html) 研究平台的一部分——面向开放视觉智能与世界建模的产学研联合枢纽。

| 章节 | 内容 |
|------|------|
| [环境配置说明](#环境配置说明) | 克隆、`.env`、checkpoint 根目录 |
| [Evaluation 环境](#evaluation-环境配置) | Conda `worldarena` 环境与 metric runtime 编译 |
| [下载 Checkpoint](#下载-checkpoint-与资产) | Metric 权重、HF/Torch 缓存、一键脚本 |
| [第三方源码](#第三方源码配置) | 子模块布局（见 `thirdparty/README.md`） |
| [模型 adapter 权重](#模型-adapter-权重) | 可选生成模型 checkpoint |
| [运行 Evaluation](#运行-evaluation) | Manifest → 生成 → 评测 |
| [Physics Simulator](#physics-simulator-runtime) | 外部 simulator 轨道 |

## 环境配置说明

#### 1. 克隆仓库

```shell
git clone --recurse-submodules https://github.com/OpenEnvision/WorldArena.git WorldArena
cd WorldArena
```

若已克隆但未拉子模块：

```shell
git submodule update --init --recursive
```

完整子模块 ↔ 模型配置对照见 [`thirdparty/README.md`](https://github.com/OpenEnvision/WorldArena/blob/main/thirdparty/README.md)。

#### 2. 配置环境路径

在运行 evaluation 或模型 adapter 之前，在仓库根目录创建 `.env` 文件。WorldArena 实际读取的变量如下：

```sh
# 仅当 ckpt/ 不在 checkout 内时需要
WORLDARENA_CHECKPOINT_ROOT=/path/to/ckpt

# 建议使用已验证的 CUDA 12.1 栈
CUDA_HOME=/path/to/cuda-12.1

# 可选：外部 physics simulator 运行时
WORLDARENA_PHYSICS_PYTHON_BIN=/path/to/physics-pipeline-env/bin/python
```

`WORLDARENA_CHECKPOINT_ROOT` 是唯一的 checkpoint 根目录。未设置时，所有 checkpoint 类路径都解析到 `WorldArena/ckpt`；设置后必须保持相同的目录布局（`huggingface/`、`torch/`、`official_metrics/`、各模型目录等）。

建议同时导出 HF/Torch 缓存（`worldarena` CLI 也会自动应用）：

```sh
export HF_HOME="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}/torch"
```

#### 3. 导出环境变量

创建 `.env` 后，在每个将要运行 WorldArena 的 shell 中导出变量：

!!! note "每个新终端都要重复"
    每个新终端会话都需要重新执行这一步。

```sh
set -a
source .env
set +a
export $(grep -v '^#' .env | xargs)
```

#### 4. API 访问（可选）

如果要评测 `config/models/` 中的 API 模型，把 provider key 放在本地 `.secrets` 文件或 shell 环境变量中。WorldArena 不会提交 API 凭证。

## Evaluation 环境配置

开源基线只需要一个 Conda 环境，环境名约定为 `worldarena`。不要使用旧的 `new` 环境。

#### 1. 创建并激活环境

```shell
# 已在 CUDA 12.1 上验证
export CUDA_HOME=/path/to/cuda-12.1/

conda create -n worldarena python=3.10 -y
conda activate worldarena
```

#### 2. 安装核心依赖

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

如果已装好的环境里 conda-forge solver 不能收敛，再使用 PyPI fallback：

```shell
python -m pip install decord==0.6.0
```

#### 3. 安装 official metric 运行时依赖

多个 headline 指标（`camera_error`、`reprojection_error`、`object_detection`、`motion_accuracy` 及相关 flow 指标）会通过 in-tree adapter 复用上游实现。需要完整 official benchmark 路径时，请初始化指标源码 submodule 并编译 DROID-SLAM：

WorldArena 只在 `worldarena/benchmark/official_backends/` 中保留指标
adapter。DROID-SLAM、SEA-RAFT、GroundingDINO、SAM2 和 VFIMamba 都会
直接复用 `thirdparty/` 下的 submodule checkout。

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

安装其余 native metric runtime（已注册子模块用 `submodule update`，否则手动 clone 一次）：

```shell
git submodule update --init \
  thirdparty/AMT \
  thirdparty/RAFT \
  thirdparty/Pulse-of-Motion \
  thirdparty/Depth-Anything-3 \
  thirdparty/met3r

# 尚未注册为子模块 — 按需手动 clone
git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything thirdparty/Video-Depth-Anything
git clone --depth 1 https://github.com/soCzech/TransNetV2 thirdparty/TransNetV2
git -C thirdparty/TransNetV2 lfs pull
```

可选 suite 级分布 metric（上游依赖 pin 与 baseline 环境可能冲突）：

```shell
git clone --depth 1 https://github.com/MCG-NKU/FVMD-frechet-video-motion-distance thirdparty/FVMD-frechet-video-motion-distance
git clone --depth 1 https://github.com/MCG-NKU/JEDi thirdparty/JEDi
```

`hpsv3_norm` 需在 `worldarena` 环境中安装 inferencer：

```shell
python -m pip install hpsv3
```

可选 MegaSAM `camera_error` backend（使用独立 conda 环境）：

```shell
export MEGASAM_PYTHON=/path/to/mega_sam/bin/python
```

#### 4. 验证环境

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

当前已配置并验证过的 evaluation 环境使用 `decord==0.6.0`、`pytorch3d==0.7.8`、`pyiqa==0.1.15.post2`、`open3d==0.19.0`、`xformers==0.0.28.post3`、`torch-scatter==2.1.2+pt25cu121`、`causal-conv1d==1.5.0.post8`、`mamba-ssm==2.2.4`。

如果从 PyPI 安装后 `pip check` 只报 `decord 0.6.0 is not supported on this platform`，通常是 PyPI wheel 元数据仍标成 CPython 3.6；运行时一般可以正常 import。开源环境优先使用 conda-forge `decord`，否则单独记录这个 `decord` 元数据问题，并用 `python -c "import decord; print(decord.__version__)"` 验证。

## 下载 Checkpoint 与资产

以下命令默认在 WorldArena checkout 内执行；若使用共享大盘布局，请确保 `WORLDARENA_CHECKPOINT_ROOT` 指向 active checkpoint root。

!!! tip "一键下载"
    ```shell
    bash scripts/download_metric_checkpoints.sh          # 核心 metric + official runtime 权重
    bash scripts/download_metric_checkpoints.sh --all    # + HPSv3、JEDi、TransNetV2、HF/Torch 缓存
    ```

评测时 checkpoint 缺失会直接 `FileNotFoundError` — WorldArena 不会在打分阶段在线下载权重。

### Metric ↔ checkpoint 对照表

| 指标 | `ckpt/` 路径 | 下载来源 |
|------|-------------|----------|
| `depth_collision` | `Video-Depth-Anything/metric_video_depth_anything_vits.pth` | [HF: depth-anything/Metric-Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small) |
| `long_sequence_*`、`long_horizon_diagnostics` | `amt_model/amt-s.pth`、`raft_model/models/raft-things.pth`、`musiq/musiq_spaq_ckpt-358bb6af.pth` | [HF: lalala125/AMT](https://huggingface.co/lalala125/AMT)、[RAFT Dropbox](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip)、[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth) |
| `temporal_calibration`、`physical_fps_*` | `Pulse-of-Motion/vc_common_10_60fps.ckpt` | [HF: xiangbog/Visual_Chronometer](https://huggingface.co/xiangbog/Visual_Chronometer) |
| `camera_error`、`reprojection_error`（DROID） | `official_metrics/droid.pth` | [Google Drive（gdown）](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh) |
| `object_detection`、`motion_accuracy` | `official_metrics/groundingdino_swint_ogc.pth`、`sam_vit_h_4b8939.pth`、`sam2.1_hiera_base_plus.pt`、`bert-base-uncased/` | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)、[SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)、[HF: sam2.1-hiera-base-plus](https://huggingface.co/facebook/sam2.1-hiera-base-plus)、[HF: bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| `optical_flow_aepe`、`motion_magnitude`、`motion_accuracy` | `official_metrics/Tartan-C-T-TSKH-spring540x960-M.pth`、`official_metrics/models/raft-things.pth` | [HF: Yarimasune/SEA-RAFT](https://huggingface.co/Yarimasune/SEA-RAFT)，从 `raft_model/` 复制 |
| `motion_smoothness` | `official_metrics/VFIMamba.pkl` | [HF: MCG-NJU/VFIMamba](https://huggingface.co/MCG-NJU/VFIMamba)（`model.pkl` 复制为 `VFIMamba.pkl`） |
| `long_sequence_aesthetic_quality` | `official_metrics/sac+logos+ava1-l14-linearMSE.pth` | [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth) |
| `depth_accuracy` | `huggingface/hub` → `depth-anything/Depth-Anything-V2-Small-hf` | [HF 模型页](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| `reconstruction_consistency`、`met3r_consistency` | `DA3-GIANT/`、HF `naver/MASt3R_*`、`torch/hub` DINO | [HF: DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT)、[HF: MASt3R](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric) |
| `semantic_alignment`、`background_consistency`、`memory_revisit_consistency` | HF `openai/clip-vit-base-patch16`、`facebook/dinov2-base` | [CLIP](https://huggingface.co/openai/clip-vit-base-patch16)、[DINOv2](https://huggingface.co/facebook/dinov2-base) |
| `hpsv3_norm` | `HPSv3/HPSv3.safetensors`、`HPSv3/HPSv3_7B_local.yaml`、`Qwen2-VL-7B-Instruct/` | [HF: MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3)、[配置模板](https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml)、[HF: Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| `segment_continuity` | `TransNetV2/transnetv2-pytorch-weights.pth` + `thirdparty/TransNetV2/inference-pytorch/` | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) |
| `jedi`（video suite） | `JEDi/vith16.pth.tar`、`JEDi/ssv2-probe.pth.tar` | [Meta JEPA](https://dl.fbaipublicfiles.com/jepa/vith16/) |

In-tree adapter 会在评测时直接从 `official_metrics/` 解析文件，文件名须与上表一致。

#### 1. Benchmark metric 权重

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

#### 2. Official metric 运行时权重

这些权重支撑 official camera/flow 指标（`object_detection`、`motion_accuracy`、camera/flow 指标）：

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

#### 3. 质量 / 分布类 metric 额外权重

运行 `hpsv3_norm`、`segment_continuity` 或 video suite 的 `jedi` 时需要：

```shell
CKPT="${WORLDARENA_CHECKPOINT_ROOT:-ckpt}"

# HPSv3 奖励模型 + Qwen2-VL backbone
wget -q -O "$CKPT/HPSv3/HPSv3.safetensors" \
  https://huggingface.co/MizzenAI/HPSv3/resolve/main/HPSv3.safetensors
wget -q -O "$CKPT/HPSv3/HPSv3_7B_local.yaml" \
  https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir "$CKPT/Qwen2-VL-7B-Instruct"

# JEDi / V-JEPA（视频分布 metric）
wget -q -O "$CKPT/JEDi/vith16.pth.tar" \
  https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
wget -q -O "$CKPT/JEDi/ssv2-probe.pth.tar" \
  https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar

# TransNetV2（segment_continuity）：克隆上游、拉 LFS 权重、转换为 PyTorch
git clone --depth 1 https://github.com/soCzech/TransNetV2 thirdparty/TransNetV2
git -C thirdparty/TransNetV2 lfs pull
cd thirdparty/TransNetV2/inference-pytorch && python convert_weights.py
cp transnetv2-pytorch-weights.pth "$CKPT/TransNetV2/"
```

或使用仓库脚本：

```shell
bash scripts/download_metric_checkpoints.sh --all
```

#### 4. Hugging Face 与 Torch cache

WorldArena 把 HF/Torch cache 固定在 checkpoint root 下，并以 offline 模式运行 bundled adapter。evaluation 前请预加载所需 backend：

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

## 模型 Adapter 权重

大型模型权重是可选项。只需为实际要跑的 `config/models/` 中的 adapter 补齐 checkpoint。

| 配置 | 上游代码 | `ckpt/` 权重目录 |
|------|----------|-----------------|
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_video_dense.yaml` | `thirdparty/lingbot-video` | `lingbot-video-dense-1.3b` |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam` + Wan2.1 配套权重 |
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

示例：

```shell
huggingface-cli download acvlab/ABot-World-0-5B-LF --local-dir ckpt/ABot-World-0-5B-LF
```

!!! tip "离线 evaluation 契约"
    checkpoint 缺失时会直接抛出 `FileNotFoundError`。WorldArena 不会在 evaluation 期间回退到其他 cache root 或在线下载。

## 第三方源码配置

`thirdparty/` 只用于源码 checkout 或符号链接。checkpoint、HF cache、Torch cache 都留在 active checkpoint root 下。

#### 目录契约

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
  thirdparty/          # git 子模块 — 仅源码
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
    predictions/       # 生成视频
    benchmark/reports/ # 评测报告
```

`artifacts/` 与 `.cache/` 为可丢弃工作目录，不承担 checkpoint 职责。

#### 模型 adapter checkout

按需初始化子模块（名称须与 `config/models/*.yaml` 一致）：

```shell
git submodule update --init thirdparty/lingbot-world
git submodule update --init thirdparty/FlashWorld thirdparty/WonderWorld
```

完整列表见 [`thirdparty/README.md`](https://github.com/OpenEnvision/WorldArena/blob/main/thirdparty/README.md)。

实践建议：

- 尽量让 clone 名称与内置配置默认值保持一致
- 在本地实验记录中固定上游 commit，不要把源码提交进这里
- 每个仓库和权重发布都分别接受其上游许可证
- 不要把 checkpoint 存到 `thirdparty/` 里

已移除的路径机制：

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend 特定的 fallback 搜索链

## 运行 Evaluation

以下命令默认已执行 `conda activate worldarena` 并导出 `.env` 变量。

#### 1. 构建 benchmark manifest

```shell
worldarena build-manifest --config config/benchmark.yaml
```

只有在你明确需要 raw asset audit、且暂时不写 manifest 时，才单独运行：

```shell
worldarena build-inventory --config config/benchmark.yaml
```

#### 2. 生成 prediction 或准备已有目录

使用 bundled adapter 生成：

```shell
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

或一次性路径：

```shell
worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

期望的 prediction 布局：

```text
artifacts/predictions/<run-name>/
  image_static/<sample_id>.mp4
  image_dynamic/<sample_id>.mp4
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

#### 3. 评测 prediction

单进程 evaluation：

```shell
worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root artifacts/predictions/lingbot_world
```

常用过滤参数：

- `--suites image_static,video_dynamic`
- `--limit 32`
- `--overwrite`
- `--model-config config/models/lingbot_world.yaml`：在打分前应用模型特定的 metric override

evaluation 产物写入 `artifacts/benchmark/reports/`：

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `leaderboard.csv`

只有当 `missing_predictions = 0`，且 suites、manifest snapshot、metric config 完全一致时，runs 才可直接对比。

!!! tip "最重指标"
    `motion_accuracy` 是最重的 dynamic 指标。默认 benchmark 配置为 80 GB 卡启用了 SAM2 video/state offload。多 GPU 主机可通过 `benchmark.metric_runtime.motion_accuracy` 的 `sam2_device`、`flow_device` 拆分设备。

## Physics Simulator Runtime

simulator-backed physics track 与上面的单 `worldarena` 指标环境分开。WorldArena 仓库内代码负责构建 manifest、stage prediction video、检查 artifact contract、重建派生 worldlines，并调用外部 simulator pipeline；但仓库本身不 vendor、也不安装这个外部 pipeline。

外部 checkout 需要满足这个契约：

```text
thirdparty/physics-pipeline/
  assets/<sample_id>.mp4
  assets/<sample_id>.json
  benchmark_export.py
  benchmark_eval.py
  main_part2.py
```

运行时选择方式：

```shell
export WORLDARENA_PHYSICS_PYTHON_BIN=/path/to/physics-pipeline-env/bin/python
worldarena build-physics-manifest \
  --physics-pipeline-root thirdparty/physics-pipeline
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

如果没有设置 `WORLDARENA_PHYSICS_PYTHON_BIN`，WorldArena 会依次回退到每个 physics case metadata 里记录的 `python_bin`，再回退到当前 Python。

当前仓库侧能验证的是：

```shell
python -m pytest tests/test_physics_simulator_derived_metrics.py
```

这只验证 in-tree 派生 simulator 指标和配置注册；不能证明 `benchmark_export.py`、`main_part2.py` 或 `benchmark_eval.py` 可运行，因为这些属于外部 physics pipeline checkout。
