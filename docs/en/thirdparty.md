# Runtime Setup

WorldArena uses one checkpoint root per checkout:

- default: `WorldArena/ckpt`
- optional large-disk override: `WORLDARENA_CHECKPOINT_ROOT=/path/to/ckpt`

Everything checkpoint-related resolves under that directory: model weights, benchmark weights, reference metric weights, Hugging Face cache, Transformers cache, and Torch cache. `thirdparty/` keeps source checkouts only. `cache/` is scratch runtime output only.

## Hard Rule

Removed path mechanisms:

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend-specific checkpoint search chains

Current fixed cache locations:

- `ckpt/huggingface`
- `ckpt/huggingface/hub`
- `ckpt/transformers`
- `ckpt/torch`

WorldArena also forces offline HF/Transformers runtime mode for bundled adapters and backends. If a required checkpoint is missing, the code now raises a clear error instead of falling back to downloads or alternate roots.

## Environment Setup Order

The open-source baseline is a single Conda environment named `worldarena`. Do not use the old `new` environment.

1. Create the environment with Python 3.10.

```bash
conda create -n worldarena python=3.10 -y
conda activate worldarena
```

2. Install CUDA 12.1 PyTorch first. The tested stack is Python 3.10.14, PyTorch 2.5.1, torchvision 0.20.1, and CUDA 12.1.

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

3. Install WorldArena itself.

```bash
python -m pip install -e .
```

4. Install `decord` for video decoding. Prefer the conda-forge package on a fresh environment because it has Python 3.10 metadata.

```bash
conda install -c conda-forge decord=0.6.0 -y
```

If conda-forge cannot solve in an already-populated environment, use the PyPI fallback instead:

```bash
python -m pip install decord==0.6.0
```

5. Install benchmark and metric runtime packages.

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

6. If you evaluate adapters that use Mamba or selective scan kernels, install the CUDA-matched wheels after PyTorch is present.

```bash
python -m pip install causal-conv1d==1.5.0.post8 mamba-ssm==2.2.4
```

7. Expose `ffmpeg`. `imageio-ffmpeg` ships a static binary; linking it into the environment is enough for the tests and video IO.

```bash
ln -sfn "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
  "$CONDA_PREFIX/bin/ffmpeg"
```

8. Install third-party metric source checkouts under `thirdparty/`.

```bash
mkdir -p thirdparty
git clone https://github.com/DepthAnything/Video-Depth-Anything thirdparty/Video-Depth-Anything
git clone https://github.com/MCG-NKU/AMT thirdparty/AMT
git clone https://github.com/princeton-vl/RAFT thirdparty/RAFT
git clone https://github.com/taco-group/Visual_Chronometer thirdparty/Pulse-of-Motion
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 thirdparty/Depth-Anything-3
git clone --recurse-submodules https://github.com/mohammadasim98/met3r thirdparty/met3r
```

9. Prepare checkpoints under `ckpt/`. The canonical metric paths are listed below. Large model-adapter checkpoints are optional and should be added only for the methods you intend to run.

10. Verify the environment before launching a full evaluation.

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

The environment prepared for the current evaluation run uses `decord==0.6.0`, `pytorch3d==0.7.8`, `pyiqa==0.1.15.post2`, `open3d==0.19.0`, `xformers==0.0.28.post3`, `torch-scatter==2.1.2+pt25cu121`, `causal-conv1d==1.5.0.post8`, and `mamba-ssm==2.2.4`.

If `pip check` reports only `decord 0.6.0 is not supported on this platform` after installing from PyPI, the package is usually importable but its published wheel metadata is tagged for CPython 3.6. Prefer conda-forge `decord`; otherwise treat that single `decord` metadata warning separately and verify with `python -c "import decord; print(decord.__version__)"`.

## Physics Simulator Runtime

The simulator-backed physics track is separate from the single `worldarena` metric environment above. The in-tree WorldArena code builds manifests, stages prediction videos, validates artifact contracts, reconstructs derived worldlines, and invokes an external simulator pipeline, but it does not vendor or install that pipeline.

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

```bash
export WORLDARENA_PHYSICS_PYTHON_BIN=/path/to/physics-pipeline-env/bin/python
worldarena build-physics-manifest \
  --physics-pipeline-root thirdparty/physics-pipeline
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

If `WORLDARENA_PHYSICS_PYTHON_BIN` is unset, WorldArena falls back to the `python_bin` recorded in each physics case metadata file, then to the current Python executable. A real simulator run therefore requires the external pipeline environment to be configured separately and exposed through one of those mechanisms.

Current repository-side validation for this path is:

```bash
python -m pytest tests/test_physics_simulator_derived_metrics.py
```

This verifies the in-tree derived simulator metrics and config registration. It does not prove that `benchmark_export.py`, `main_part2.py`, or `benchmark_eval.py` can run, because those belong to the external physics pipeline checkout.

## Directory Contract

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

`cache/` remains disposable workspace output. It is no longer a checkpoint mechanism.

## Model Mapping

| Config | Source checkout | Checkpoint path(s) under `ckpt/` |
| --- | --- | --- |
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_world_base_act_preview.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-act-preview`, `generation.shared_checkpoint_dir=lingbot-world-base-cam` |
| `lingbot_world_fast.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam`, `generation.fast_checkpoint_dir=lingbot-world-fast` |
| `cosmos_predict25_14b.yaml` | `thirdparty/cosmos-predict2.5` | `Cosmos-Predict2.5-14B`, `Cosmos-Reason1-7B`, `Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth` |
| `cosmos_predict25_2b.yaml` | `thirdparty/cosmos-predict2.5` | `Cosmos-Predict2.5-2B` plus the same auxiliary text/tokenizer checkpoints as the 14B config |
| `inspatio_world.yaml` | `thirdparty/inspatio-world` | `inspatio--world` |
| `infinite_world.yaml` | `thirdparty/Infinite-World` | `MeiGen-AI--Infinite-World` |
| `neoverse.yaml` | `thirdparty/NeoVerse` | `Yuppie1204--NeoVerse` |
| `lyra1.yaml` | `thirdparty/Lyra-1` | `Lyra` plus `nvidia--GEN3C-Cosmos-7B`, `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`, `google-t5--t5-11b`, `Ruicheng--moge-vitl` |
| `lyra2.yaml` | `thirdparty/Lyra-2` | `Lyra-2.0` plus `depth-anything--DA3NESTED-GIANT-LARGE`, `google/umt5-xxl`, `Ruicheng--moge-vitl` |
| `gen3c.yaml` | `thirdparty/Lyra-1` | `nvidia--GEN3C-Cosmos-7B` plus `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`, `google-t5--t5-11b`, `Ruicheng--moge-vitl` |
| `flashworld.yaml`, `flashworld_image_static.yaml` | `thirdparty/FlashWorld` | `imlixinyang--FlashWorld` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam` plus `Wan-AI--Wan2.1-T2V-1.3B` |
| `hunyuanworld.yaml` | `thirdparty/HunyuanWorld-1.0` | `HunyuanWorld-1` plus `ZIM/zim_vit_l_2092` |
| `luciddreamer.yaml` | `thirdparty/LucidDreamer` | `stabilityai--stable-diffusion-2-inpainting` and any required upstream HF cache entries under `huggingface/` |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay`, `HY-WorldPlay/ar_distilled_action_model/model.safetensors` |
| `wonderjourney.yaml` | `thirdparty/WonderJourney` | `WonderJourney/dpt_beit_large_512.pt` plus `sd2-community--stable-diffusion-2-inpainting` and any required upstream HF cache entries under `huggingface/` |
| `wonderworld.yaml` | `thirdparty/WonderWorld` | `WonderWorld/repvit_sam.pt`, `sd2-community--stable-diffusion-2-inpainting`, `prs-eth--marigold-v1-0`, `prs-eth--marigold-normals-v0-1`, and any required upstream HF cache entries under `huggingface/` |
| `yume.yaml` | `thirdparty/YUME` | `stdstu123--Yume-5B-720P`, `InternVL3-2B-Instruct` |
| `worldfm.yaml` | `thirdparty/worldfm` | `worldfm` plus `ZIM/zim_vit_l_2092` |
| `matrix_game_1.yaml` | `thirdparty/Matrix-Game-1` | `Matrix-Game-1.0` |
| `matrix_game_2.yaml` | `thirdparty/Matrix-Game/Matrix-Game-2` | `Skywork--Matrix-Game-2.0` |
| `matrix_game.yaml` | `thirdparty/Matrix-Game-3.0` | `Matrix-Game-3.0` |
| `hunyuan_gamecraft.yaml` | `thirdparty/Hunyuan-GameCraft-1.0` | `tencent--Hunyuan-GameCraft-1.0` |

## Benchmark Runtime Weights

| Runtime item | Canonical locator under `ckpt/` |
| --- | --- |
| CLIP backend | `openai/clip-vit-base-patch16 -> models--openai--clip-vit-base-patch16/snapshots/<revision>` |
| DINOv2 backend | `facebook/dinov2-base -> models--facebook--dinov2-base/snapshots/<revision>` |
| Depth Accuracy | `models--depth-anything--Depth-Anything-V2-Small-hf` or an equivalent local `depth-anything/Depth-Anything-V2-Small-hf` snapshot |
| Depth Collision | `Video-Depth-Anything/metric_video_depth_anything_vits.pth` |
| Reconstruction Consistency | `DA3-GIANT` plus `thirdparty/Depth-Anything-3/src` |
| MET3R MASt3R backbone | `huggingface/hub/models--naver--MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` |
| MET3R DINO feature backbone | `torch/hub/facebookresearch_dino_main` and `torch/hub/checkpoints/dino_deitsmall16_pretrain.pth` |
| Long-Sequence AMT | `amt_model/amt-s.pth` |
| Long-Sequence RAFT | `raft_model/models/raft-things.pth` |
| Long-Sequence MUSIQ | `musiq/musiq_spaq_ckpt-358bb6af.pth` |
| Temporal Calibration | `Pulse-of-Motion/vc_common_10_60fps.ckpt` |
| BRISQUE / CLIP-IQA / Aesthetic quality caches | `torch/hub/pyiqa` |
| FVMD, optional suite-level | `thirdparty/FVMD-frechet-video-motion-distance` |
| JEDi, optional suite-level | `thirdparty/JEDi` plus its model config/weights |

Fixed source checkouts for bundled metric runtimes:

- `thirdparty/met3r`
- `thirdparty/Pulse-of-Motion`
- `thirdparty/AMT`
- `thirdparty/RAFT`
- `thirdparty/Depth-Anything-3`
- `thirdparty/Video-Depth-Anything`

These source locations are no longer redirected by custom root environment variables.

`fvmd` and `jedi` are suite-level distribution metrics. They are not part of the single-environment baseline above because the public PyPI packages currently pin mutually incompatible dependency ranges. Configure them only when you have a compatible external runtime under `thirdparty/FVMD-frechet-video-motion-distance` and `thirdparty/JEDi`; otherwise they should be treated as optional suite-level metrics outside the verified per-sample metric stack.

## Path Resolution Rules

- Project-relative paths resolve against the WorldArena repo root.
- Checkpoint-like paths must stay under the active checkpoint root: `WorldArena/ckpt` by default, or `WORLDARENA_CHECKPOINT_ROOT` when explicitly set.
- Adapters that support subpaths under a checkpoint tree, such as `InSpatio-World`, `WorldFM`, and `Matrix-Game-2`, resolve those subpaths inside their configured `checkpoint_dir`.
- Missing checkpoints now raise `FileNotFoundError` directly.
- WorldArena does not keep compatibility layers that try unrelated roots in sequence.

## Third-Party Repository Policy

`thirdparty/` is for source checkouts or symlinks only.

Practical rules:

- keep clone names aligned with bundled config defaults when possible
- pin upstream commits in local notes or experiment logs, not by committing source here
- accept upstream licenses separately for each repo and each weight release
- do not store checkpoints inside `thirdparty/`

## Upstream Notes

- `lingbot-world` still supports `generation.shared_checkpoint_dir` and `generation.fast_checkpoint_dir`, but both must now be explicit `ckpt/...` paths. The runner materializes the upstream expected layout from those explicit local directories.
- `InSpatio-World` keeps upstream subdirectory names such as `InSpatio-World-1.3B/`, `DA3/`, and optional `taehv/` inside `ckpt/inspatio--world`.
- `HunyuanWorld` still expects its auxiliary source tree layout around `thirdparty/HunyuanWorld-1.0`, including the `ZIM`, `Real-ESRGAN`, and `MoGe` source checkouts. The ZIM checkpoint itself is now staged from `ckpt/ZIM/zim_vit_l_2092` at runtime instead of living under `thirdparty/`.
- `WonderJourney` and `WonderWorld` follow the benchmark output contract: the benchmark prediction is a rendered-frame `.mp4`; zip archives, when present, are sidecars for generated frames and 3D assets only.
- `LucidDreamer`, `WonderJourney`, and `WonderWorld` still contain upstream repo-id based loading paths. WorldArena now pins those caches to `ckpt/` and runs offline, so preload the required HF assets into the fixed cache tree under `ckpt/` before evaluation.
- `WorldCam`, `Lyra-1`, `Lyra-2`, and `GEN3C` require companion checkpoints under `ckpt/` beyond their primary `checkpoint_dir`.
