# 运行时布局

WorldArena 每个 checkout 只使用一个 checkpoint 根目录：

- 默认：`WorldArena/ckpt`
- 大盘部署可选覆盖：`WORLDARENA_CHECKPOINT_ROOT=/path/to/ckpt`

所有 checkpoint 相关内容都统一落在这里：模型权重、benchmark 运行时权重、参考指标权重、Hugging Face cache、Transformers cache、Torch cache。`thirdparty/` 只保留源码 checkout，`cache/` 只保留临时运行产物。

## 硬约束

以下路径机制已经移除：

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend 各自维护的 checkpoint 搜索链

当前固定缓存位置：

- `ckpt/huggingface`
- `ckpt/huggingface/hub`
- `ckpt/transformers`
- `ckpt/torch`

内置 adapter 和 backend 还会强制开启 HF/Transformers offline 模式。如果某个 checkpoint 缺失，代码现在会直接报清晰错误，不再回退到下载或其他目录。

## 环境配置顺序

开源基线只需要一个 Conda 环境，环境名约定为 `worldarena`。不要使用旧的 `new` 环境。

1. 创建 Python 3.10 环境。

```bash
conda create -n worldarena python=3.10 -y
conda activate worldarena
```

2. 先安装 CUDA 12.1 对应的 PyTorch。当前验证过的组合是 Python 3.10.14、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.1。

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

3. 安装 WorldArena 本体。

```bash
python -m pip install -e .
```

4. 安装视频解码依赖 `decord`。新环境优先使用 conda-forge 包，它带有 Python 3.10 元数据。

```bash
conda install -c conda-forge decord=0.6.0 -y
```

如果已装好的环境里 conda-forge solver 不能收敛，再使用 PyPI fallback：

```bash
python -m pip install decord==0.6.0
```

5. 安装 benchmark 和 metric runtime 依赖。

```bash
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
```

6. 如果要评测依赖 Mamba 或 selective scan kernel 的模型 adapter，在 PyTorch 装好后再装这些 CUDA 匹配包。

```bash
python -m pip install causal-conv1d==1.5.0.post8 mamba-ssm==2.2.4
```

7. 暴露 `ffmpeg`。`imageio-ffmpeg` 自带静态二进制，链接到当前 Conda 环境即可满足测试和视频 IO。

```bash
ln -sfn "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
  "$CONDA_PREFIX/bin/ffmpeg"
```

8. 把指标运行时源码 checkout 放到 `thirdparty/`。

```bash
mkdir -p thirdparty
git clone https://github.com/DepthAnything/Video-Depth-Anything thirdparty/Video-Depth-Anything
git clone https://github.com/MCG-NKU/AMT thirdparty/AMT
git clone https://github.com/princeton-vl/RAFT thirdparty/RAFT
git clone https://github.com/taco-group/Visual_Chronometer thirdparty/Pulse-of-Motion
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 thirdparty/Depth-Anything-3
git clone --recurse-submodules https://github.com/mohammadasim98/met3r thirdparty/met3r
```

9. 按下面的规范路径准备 `ckpt/`。大型模型 adapter 的 checkpoint 是可选项，只需要为实际要跑的方法补齐。

10. 在正式 evaluation 前做一次环境验证。

```bash
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

## Physics Simulator Runtime

simulator-backed physics track 和上面的单 `worldarena` 指标环境是分开的。WorldArena 仓库内代码负责构建 manifest、stage prediction video、检查 artifact contract、重建派生 worldlines，并调用外部 simulator pipeline；但仓库本身不 vendor、也不安装这个外部 pipeline。

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

```bash
export WORLDARENA_PHYSICS_PYTHON_BIN=/path/to/physics-pipeline-env/bin/python
worldarena build-physics-manifest \
  --physics-pipeline-root thirdparty/physics-pipeline
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

如果没有设置 `WORLDARENA_PHYSICS_PYTHON_BIN`，WorldArena 会依次回退到每个 physics case metadata 里记录的 `python_bin`，再回退到当前 Python。因此真正跑 simulator 评测时，外部 pipeline 环境必须单独配置，并通过上述机制暴露给 WorldArena。

当前仓库侧能验证的是：

```bash
python -m pytest tests/test_physics_simulator_derived_metrics.py
```

这只验证 in-tree 派生 simulator 指标和配置注册；不能证明 `benchmark_export.py`、`main_part2.py` 或 `benchmark_eval.py` 可运行，因为这些属于外部 physics pipeline checkout。

## 目录契约

```text
WorldArena/
  ckpt/
  thirdparty/
    lingbot-world/
    inspatio-world/
    Infinite-World/
    NeoVerse/
    Lyra-1/
    Lyra-2/
    FlashWorld/
    WorldCam/
    HunyuanWorld-1.0/
    LucidDreamer/
    HY-WorldPlay/
    WonderJourney/
    WonderWorld/
    YUME/
    worldfm/
    Hunyuan-GameCraft-1.0/
    Matrix-Game-1/
    Matrix-Game/
    Matrix-Game-3.0/
    Video-Depth-Anything/
    met3r/
    AMT/
    RAFT/
    Pulse-of-Motion/
    Depth-Anything-3/
  cache/
```

`cache/` 仍然只是可丢弃的工作目录，不再承担 checkpoint 职责。

## 模型映射

| 配置 | 源码 checkout | `ckpt/` 下的 checkpoint 路径 |
| --- | --- | --- |
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_world_base_act_preview.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-act-preview`，`generation.shared_checkpoint_dir=lingbot-world-base-cam` |
| `lingbot_world_fast.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam`，`generation.fast_checkpoint_dir=lingbot-world-fast` |
| `cosmos_predict25_14b.yaml` | `thirdparty/cosmos-predict2.5` | `Cosmos-Predict2.5-14B`、`Cosmos-Reason1-7B`、`Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth` |
| `cosmos_predict25_2b.yaml` | `thirdparty/cosmos-predict2.5` | `Cosmos-Predict2.5-2B`，以及与 14B 配置相同的辅助 text/tokenizer checkpoint |
| `inspatio_world.yaml` | `thirdparty/inspatio-world` | `inspatio--world` |
| `infinite_world.yaml` | `thirdparty/Infinite-World` | `MeiGen-AI--Infinite-World` |
| `neoverse.yaml` | `thirdparty/NeoVerse` | `Yuppie1204--NeoVerse` |
| `lyra1.yaml` | `thirdparty/Lyra-1` | `Lyra`，以及 `nvidia--GEN3C-Cosmos-7B`、`nvidia--Cosmos-Tokenize1-CV8x8x8-720p`、`google-t5--t5-11b`、`Ruicheng--moge-vitl` |
| `lyra2.yaml` | `thirdparty/Lyra-2` | `Lyra-2.0`，以及 `depth-anything--DA3NESTED-GIANT-LARGE`、`google/umt5-xxl`、`Ruicheng--moge-vitl` |
| `gen3c.yaml` | `thirdparty/Lyra-1` | `nvidia--GEN3C-Cosmos-7B`，以及 `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`、`google-t5--t5-11b`、`Ruicheng--moge-vitl` |
| `flashworld.yaml`、`flashworld_image_static.yaml` | `thirdparty/FlashWorld` | `imlixinyang--FlashWorld` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam`，以及 `Wan-AI--Wan2.1-T2V-1.3B` |
| `hunyuanworld.yaml` | `thirdparty/HunyuanWorld-1.0` | `HunyuanWorld-1`，以及 `ZIM/zim_vit_l_2092` |
| `luciddreamer.yaml` | `thirdparty/LucidDreamer` | `stabilityai--stable-diffusion-2-inpainting`，以及 `huggingface/` 下的上游缓存条目 |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay`，`HY-WorldPlay/ar_distilled_action_model/model.safetensors` |
| `wonderjourney.yaml` | `thirdparty/WonderJourney` | `WonderJourney/dpt_beit_large_512.pt`，以及 `sd2-community--stable-diffusion-2-inpainting` 和 `huggingface/` 下的上游缓存条目 |
| `wonderworld.yaml` | `thirdparty/WonderWorld` | `WonderWorld/repvit_sam.pt`、`sd2-community--stable-diffusion-2-inpainting`、`prs-eth--marigold-v1-0`、`prs-eth--marigold-normals-v0-1`，以及 `huggingface/` 下的上游缓存条目 |
| `yume.yaml` | `thirdparty/YUME` | `stdstu123--Yume-5B-720P`，`InternVL3-2B-Instruct` |
| `worldfm.yaml` | `thirdparty/worldfm` | `worldfm`，以及 `ZIM/zim_vit_l_2092` |
| `matrix_game_1.yaml` | `thirdparty/Matrix-Game-1` | `Matrix-Game-1.0` |
| `matrix_game_2.yaml` | `thirdparty/Matrix-Game/Matrix-Game-2` | `Skywork--Matrix-Game-2.0` |
| `matrix_game.yaml` | `thirdparty/Matrix-Game-3.0` | `Matrix-Game-3.0` |
| `hunyuan_gamecraft.yaml` | `thirdparty/Hunyuan-GameCraft-1.0` | `tencent--Hunyuan-GameCraft-1.0` |

## Benchmark 运行时权重

| 运行时项 | `ckpt/` 下的规范定位 |
| --- | --- |
| CLIP backend | `openai/clip-vit-base-patch16 -> models--openai--clip-vit-base-patch16/snapshots/<revision>` |
| DINOv2 backend | `facebook/dinov2-base -> models--facebook--dinov2-base/snapshots/<revision>` |
| Depth Accuracy | `models--depth-anything--Depth-Anything-V2-Small-hf`，或等价的本地 `depth-anything/Depth-Anything-V2-Small-hf` snapshot |
| Depth Collision | `Video-Depth-Anything/metric_video_depth_anything_vits.pth` |
| Reconstruction Consistency | `DA3-GIANT`，并配合 `thirdparty/Depth-Anything-3/src` |
| MET3R MASt3R backbone | `huggingface/hub/models--naver--MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` |
| MET3R DINO feature backbone | `torch/hub/facebookresearch_dino_main` 和 `torch/hub/checkpoints/dino_deitsmall16_pretrain.pth` |
| Long-Sequence AMT | `amt_model/amt-s.pth` |
| Long-Sequence RAFT | `raft_model/models/raft-things.pth` |
| Long-Sequence MUSIQ | `musiq/musiq_spaq_ckpt-358bb6af.pth` |
| Temporal Calibration | `Pulse-of-Motion/vc_common_10_60fps.ckpt` |
| BRISQUE / CLIP-IQA / Aesthetic quality cache | `torch/hub/pyiqa` |
| FVMD，可选 suite-level | `thirdparty/FVMD-frechet-video-motion-distance` |
| JEDi，可选 suite-level | `thirdparty/JEDi`，以及它自己的模型配置和权重 |

内置指标运行时使用的固定源码 checkout：

- `thirdparty/met3r`
- `thirdparty/Pulse-of-Motion`
- `thirdparty/AMT`
- `thirdparty/RAFT`
- `thirdparty/Depth-Anything-3`
- `thirdparty/Video-Depth-Anything`

这些源码位置不再通过自定义 root 环境变量重定向。

`fvmd` 和 `jedi` 是 suite-level 分布指标，不属于上面这个单环境基线。公开 PyPI 包目前有互相冲突的强 pin 依赖：`fvmd` 和 `videojedi` 对 `scikit-learn` 等版本要求不一致。只有在你已经准备好兼容的外部 runtime，并把它们放在 `thirdparty/FVMD-frechet-video-motion-distance` 和 `thirdparty/JEDi` 下时，才启用这两个指标；否则它们应视为 verified per-sample metric stack 之外的可选 suite-level 指标。

## 路径解析规则

- project-relative 路径都相对于 WorldArena 仓库根目录解析。
- 所有 checkpoint 类路径都必须留在 active checkpoint root 下：默认是 `WorldArena/ckpt`，显式设置时是 `WORLDARENA_CHECKPOINT_ROOT`。
- 对于支持 checkpoint 树内子路径的 adapter，例如 `InSpatio-World`、`WorldFM`、`Matrix-Game-2`，相关子路径会在各自的 `checkpoint_dir` 内解析。
- checkpoint 缺失时直接抛 `FileNotFoundError`。
- WorldArena 不再保留“先试 A，再试 B，再试 C”的无关根目录兼容层。

## 第三方源码策略

`thirdparty/` 只用于源码 checkout 或符号链接。

实践建议：

- 尽量让 clone 名称与内置配置默认值保持一致
- 在本地实验记录中固定上游 commit，不要把源码提交进这里
- 每个仓库和权重发布都分别接受其上游许可证
- 不要把 checkpoint 存到 `thirdparty/` 里

## 上游特殊说明

- `lingbot-world` 仍然支持 `generation.shared_checkpoint_dir` 和 `generation.fast_checkpoint_dir`，但它们现在只能是显式的 `ckpt/...` 路径；runner 会基于这些显式目录物化上游所需布局。
- `InSpatio-World` 会在 `ckpt/inspatio--world` 内保留上游子目录名，例如 `InSpatio-World-1.3B/`、`DA3/` 和可选的 `taehv/`。
- `HunyuanWorld` 仍然要求 `thirdparty/HunyuanWorld-1.0` 周围保留上游辅助源码布局，包括 `ZIM`、`Real-ESRGAN`、`MoGe` 这些源码 checkout。ZIM checkpoint 本身则在运行时从 `ckpt/ZIM/zim_vit_l_2092` 映射进去，不再存放在 `thirdparty/` 下。
- `WonderJourney` 和 `WonderWorld` 都遵循 benchmark 输出契约：benchmark prediction 是由生成帧写出的 `.mp4`；如果出现 zip，只作为生成帧和 3D 资产的 sidecar。
- `LucidDreamer`、`WonderJourney` 和 `WonderWorld` 的部分上游代码仍然会走 repo-id 型加载。WorldArena 现在只把这些缓存固定到 `ckpt/` 并离线运行，因此评测前需要先把相应 HF 资产预加载到 `ckpt/` 下的固定缓存树。
- `WorldCam`、`Lyra-1`、`Lyra-2`、`GEN3C` 除了主 `checkpoint_dir` 外，还需要额外的 companion checkpoint 一并放在 `ckpt/` 下。
