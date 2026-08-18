<div align="center">

# WorldArena

**A Unified, Extensible Benchmark for World Model Evaluation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-OpenEnvision%2FWorldArena-181717.svg)](https://github.com/OpenEnvision/WorldArena)

[Paper]() · [OpenEnvision](https://openenvision.github.io/index.html) · [Hugging Face]() · [Leaderboard]()

<br>

*Is your world model a consistent, controllable, and physically plausible scene renderer?*

</div>

---

## Table of Contents

- [Overview](#overview)
- [News](#news)
- [Benchmark Design](#benchmark-design)
- [Installation](#installation)
- [Path & Environment Configuration](#path--environment-configuration)
- [Checkpoint Preparation](#checkpoint-preparation)
- [Dataset Preparation](#dataset-preparation)
- [File Formats & Contracts](#file-formats--contracts)
- [Usage](#usage)
- [Integrating Your Model](#integrating-your-model)
- [Supported Models](#supported-models)
- [Documentation & Site](#documentation--site)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

---

## Overview

**WorldArena** (WorldAtlas Arena) is a research benchmark and evaluation toolkit for **world models** — models that generate or roll out interactive 3D/4D scenes from images, videos, text, camera trajectories, or actions.

The repository provides:

| Component | Description |
|-----------|-------------|
| **4 formal benchmark suites** | `image_static`, `image_dynamic`, `video_static`, `video_dynamic` |
| **6 evaluation dimensions** | Action control, 3D/4D consistency, physics, quality, long sequence, real time |
| **36+ automatic metrics** | Geometry, flow, depth, perceptual quality, physics, long-horizon diagnostics, and more |
| **25+ model adapters** | Unified YAML contract for local repos, Python modules, and HTTP APIs |
| **Reproducible reporting** | Per-sample JSONL, per-metric summaries, dimension aggregates, leaderboard CSV |

---

## News

- **[2026/07]** Initial public release: evaluation core, model adapters, benchmark configs, and documentation site.
- **[2026/07]** Added `perceptual_quality` and expanded long-horizon diagnostics across formal suites.
- **[2026/07]** Published 25+ adapter configs under `config/models/`.

---

## Benchmark Design

### Formal suites

| Suite | Conditioning input | Typical control signal | Example tasks |
|-------|-------------------|------------------------|---------------|
| `image_static` | Single image | Camera path / pose, text prompt | Novel-view synthesis, camera-controlled rollout |
| `image_dynamic` | Single image + mask / motion label | Dynamic prompt, object motion | Object motion, scene evolution from a still frame |
| `video_static` | Video prefix | Camera trajectory | Video continuation with camera control |
| `video_dynamic` | Video prefix | Instruction / dynamic prompt | Interactive world continuation |

An additional **`experimental`** slice exists for supplemental metrics (e.g. physics-judge probes). It is declared separately in `config/benchmark.yaml` and does **not** contribute to formal headline aggregates.

### Evaluation dimensions (selected metrics)

| Dimension | Representative metrics | What it measures |
|-----------|------------------------|------------------|
| **Action control** | `camera_error`, `motion_accuracy`, `prompt_alignment` | Following camera trajectories, prompts, and motion instructions |
| **3D/4D consistency** | `reprojection_error`, `optical_flow_aepe`, `depth_collision`, `met3r_consistency`, `reconstruction_consistency` | Geometric and photometric self-consistency |
| **Physics** | `pmf`, `fvmd`, `motion_magnitude`, `motion_smoothness`, `physics_judge_*` | Physical plausibility of motion and interactions |
| **Quality** | `perceptual_quality`, `style_consistency`, `image_quality`, `hpsv3_norm`, `sharpness_retention` | Visual fidelity and artifact severity |
| **Long sequence** | `seen_region_preservation`, `object_permanence`, `memory_revisit_consistency`, `scene_memory_f1` | Memory and stability over long rollouts |
| **Real time** | `physical_fps_error`, `action_response_latency`, `latency_budget_pass_rate` | Interactive latency and throughput |

> Full metric definitions, normalization rules, and suite assignments: [`docs/en/benchmark/metrics/`](docs/en/benchmark/metrics/index.md) (also mirrored in Chinese under `docs/zh/`).

---

## Installation

### Prerequisites

| Requirement | Recommended version | Notes |
|-------------|---------------------|-------|
| OS | Linux (x86_64) | Primary development and test platform |
| Python | **3.10** | Core `worldarena` environment |
| CUDA | **12.1** (tested) | Metric backends and several adapters |
| PyTorch | **2.5.1+cu121** | See optional stack below |
| ffmpeg | any recent | Video decode / encode in metrics and adapters |
| Git + Git LFS | — | For submodules and large assets |
| Node.js | **18+** | Documentation site build only |

### 1. Clone the repository

```bash
git clone https://github.com/OpenEnvision/WorldArena.git
cd WorldArena

# Optional: initialize third-party model/metric submodules
git submodule update --init --recursive
```

### 2. Create the core evaluation environment

The **`worldarena`** Conda environment runs inventory building, metric evaluation, and lightweight adapters.

```bash
export CUDA_HOME=/path/to/cuda-12.1   # adjust to your install

conda create -n worldarena python=3.10 -y
conda activate worldarena

# PyTorch (tested stack)
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Editable install of WorldArena
python -m pip install -e .

# Additional metric / media dependencies (recommended for full evaluation)
conda install -c conda-forge decord=0.6.0 -y

python -m pip install \
  transformers==4.46.3 diffusers accelerate pyiqa open3d iopath pycocotools \
  pytorch-lightning torchmetrics timm==0.4.12 xformers==0.0.28.post3 \
  easydict roma jaxtyping lpips "pyglet<2" e3nn moviepy==1.0.3 \
  plyfile pillow-heif pycolmap imageio-ffmpeg OpenEXR kornia gsplat trimesh

python -m pip install \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html \
  pytorch3d
```

Verify the CLI:

```bash
worldarena --help
# alias: worldatlas-arena
```

### 3. Model-specific runtimes (optional)

Each upstream world model typically needs its **own** Python environment (CUDA version, custom extensions, pinned deps). WorldArena does not bundle all of these into one environment — follow each upstream repo's official install instructions.

| Model family | Upstream checkout | Notes |
|--------------|-------------------|-------|
| All repos | `git submodule update --init --recursive` — see [`thirdparty/README.md`](thirdparty/README.md) | — |
| LingBot-World | `thirdparty/lingbot-world` | upstream `requirements/` / README |
| LingBot-Video | `thirdparty/lingbot-video` | upstream README |
| CamI2V / RealCam-I2V | `thirdparty/CamI2V`, `thirdparty/RealCam-I2V` | optional: [`scripts/launch_camera_i2v_8gpu.sh`](scripts/launch_camera_i2v_8gpu.sh) |
| ABot-World | `thirdparty/ABot-World` | upstream README |
| Matrix-Game 3.5 | `bash scripts/setup_matrix_game_35.sh` | installs from upstream `requirements.txt` + downloads weights |
| Open Dreamer | `bash scripts/setup_open_dreamer.sh` | JAX/CUDA 12 stack built with upstream `uv sync` + downloads weights |

Point `python_bin` in the model YAML at the interpreter of that dedicated environment, or export `<MODEL>_PYTHON_BIN` in launch scripts.

Detailed runtime notes: [`docs/en/thirdparty.md`](docs/en/thirdparty.md).

---

## Path & Environment Configuration

WorldArena resolves **all paths relative to the repository root** unless an absolute path or environment override is provided.

### Recommended directory layout

```text
WorldArena/                          # repository root (WORLD_ROOT)
├── ckpt/                            # default checkpoint root
│   ├── huggingface/                 # HF_HOME / hub cache
│   ├── torch/                       # TORCH_HOME
│   ├── transformers/              # legacy transformers cache (optional)
│   └── <model-name>/                # per-model weight trees
├── data/                            # benchmark media (user-provided)
│   ├── image/
│   │   ├── static/
│   │   └── dynamic/
│   └── video/
│       ├── static/
│       └── dynamic/
├── thirdparty/                      # git submodules (upstream models + metrics)
│   └── README.md                    # submodule index and init commands
├── artifacts/                       # generated outputs (gitignored)
│   ├── predictions/                 # model-generated videos
│   └── benchmark/
│       ├── inventory/
│       ├── manifest/
│       └── reports/                 # evaluation JSONL / leaderboard
├── config/                          # YAML contracts
└── .env                             # local path overrides (not committed)
```

### Environment variables

Create a `.env` file at the repository root (see [`.gitignore`](.gitignore); never commit secrets):

```bash
# ── Checkpoint & cache ─────────────────────────────────────────────
# When weights live outside the repo, point here. Layout must mirror ckpt/.
WORLDARENA_CHECKPOINT_ROOT=/path/to/ckpt

HF_HOME=/path/to/ckpt/huggingface
HUGGINGFACE_HUB_CACHE=/path/to/ckpt/huggingface/hub
TRANSFORMERS_CACHE=/path/to/ckpt/huggingface/hub
TORCH_HOME=/path/to/ckpt/torch

# Offline mode (optional, after caches are populated)
# HF_HUB_OFFLINE=1
# TRANSFORMERS_OFFLINE=1

# ── Benchmark data roots ─────────────────────────────────────────
# Used by config/benchmark.yaml paths and the static site server
WORLDARENA_DATA_ROOT=/path/to/data
WORLDARENA_IMAGE_DATA=/path/to/data/image
WORLDARENA_VIDEO_DATA=/path/to/data/video
WORLDARENA_VIDEO_SPLIT_DATA=/path/to/data/video_vipe_ann
WORLDARENA_PHYSICS_DATA=/path/to/data/physics

# ── CUDA / build ───────────────────────────────────────────────────
CUDA_HOME=/path/to/cuda-12.1

# ── Legacy path remapping (optional) ───────────────────────────────
# Remaps stale absolute paths inside old manifests to a new workspace root
WORLDARENA_LEGACY_WORKSPACE_ROOT=/path/to/old/workspace

# ── Model interpreter overrides (examples) ───────────────────────
# WORLDARENA_PYTHON_BIN=python
# LINGBOT_WORLD_PYTHON_BIN=/path/to/env/bin/python
# ALAYAWORLD_PYTHON_BIN=/path/to/env/bin/python

# ── API models (optional) ──────────────────────────────────────────
# KLING_API_KEY=...
# OPENAI_API_KEY=...          # physics judge / optional backends
```

Load before every session:

```bash
set -a
source .env
set +a
```

### Config files to edit for your machine

| File | Purpose | Default paths |
|------|---------|---------------|
| [`config/benchmark.yaml`](config/benchmark.yaml) | Suite definitions, metric lists, benchmark data roots | `data/image`, `data/video` |
| [`config/datasets.yaml`](config/datasets.yaml) | Dataset analytics scanners | Update `collections[].root` to your media trees |
| [`config/models/*.yaml`](config/models/) | Per-model runtime (`repo_root`, `checkpoint_dir`, `python_bin`) | Relative to repo / `ckpt/` |

> **Tip:** Keep machine-specific overrides in `config/models/*_local.yaml` or `config/datasets.cpfs.yaml` — these patterns are listed in [`.gitignore`](.gitignore) and will not be committed.

---

## Checkpoint Preparation

### Default layout under `ckpt/`

When `WORLDARENA_CHECKPOINT_ROOT` is unset, WorldArena uses **`WorldArena/ckpt/`** as the canonical root. The helper `worldarena.common.checkpoints.checkpoint_root()` creates subfolders on demand.

```bash
mkdir -p ckpt/huggingface/hub ckpt/torch ckpt/official_metrics
```

Typical contents:

| Subdirectory | Purpose |
|--------------|---------|
| `ckpt/huggingface/` | Hugging Face Hub cache (`HF_HOME`) |
| `ckpt/torch/` | Torch hub / torchvision weights |
| `ckpt/official_metrics/` | Metric backbone weights used by the official runtime |
| `ckpt/<model>/` | Upstream checkpoint trees referenced in model YAML |

### Metric evaluation checkpoints

Benchmark metrics read weights from the same `ckpt/` root (or `WORLDARENA_CHECKPOINT_ROOT`). Missing files raise `FileNotFoundError` at evaluation time — there is no online fallback.

**Quick start** — download the bundled-metric and official-runtime weights:

```bash
bash scripts/download_metric_checkpoints.sh          # core metric weights
bash scripts/download_metric_checkpoints.sh --all    # + HPSv3, JEDi, TransNetV2, HF/Torch caches
```

Full commands and environment notes: [`docs/en/thirdparty.md`](docs/en/thirdparty.md#download-checkpoints-and-assets) (Chinese: [`docs/zh/thirdparty.md`](docs/zh/thirdparty.md)).

#### Metric ↔ checkpoint quick reference

| Metric(s) | Path under `ckpt/` | Download source |
|-----------|-------------------|-----------------|
| `depth_collision` | `Video-Depth-Anything/metric_video_depth_anything_vits.pth` | [HF: depth-anything/Metric-Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small) |
| `long_sequence_*`, `long_horizon_diagnostics` | `amt_model/amt-s.pth`, `raft_model/models/raft-things.pth`, `musiq/musiq_spaq_ckpt-358bb6af.pth` | [HF: lalala125/AMT](https://huggingface.co/lalala125/AMT), [RAFT Dropbox models.zip](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip), [IQA-PyTorch release](https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth) |
| `temporal_calibration`, `physical_fps_*` | `Pulse-of-Motion/vc_common_10_60fps.ckpt` | [HF: xiangbog/Visual_Chronometer](https://huggingface.co/xiangbog/Visual_Chronometer) |
| `camera_error`, `reprojection_error` (DROID) | `official_metrics/droid.pth` | [Google Drive via gdown](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh) |
| `motion_accuracy` | `official_metrics/groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth`, `sam2.1_hiera_base_plus.pt`, `bert-base-uncased/` | [GroundingDINO release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth), [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), [HF: facebook/sam2.1-hiera-base-plus](https://huggingface.co/facebook/sam2.1-hiera-base-plus), [HF: google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| `optical_flow_aepe`, `motion_magnitude`, `motion_accuracy` | `official_metrics/Tartan-C-T-TSKH-spring540x960-M.pth`, `official_metrics/models/raft-things.pth` | [HF: Yarimasune/SEA-RAFT](https://huggingface.co/Yarimasune/SEA-RAFT), copy from `raft_model/` |
| `motion_smoothness` | `official_metrics/VFIMamba.pkl` | [HF: MCG-NJU/VFIMamba](https://huggingface.co/MCG-NJU/VFIMamba) (`model.pkl` → rename) |
| `long_sequence_aesthetic_quality` (CLIP aesthetic head) | `official_metrics/sac+logos+ava1-l14-linearMSE.pth` | [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth) |
| `depth_accuracy` | `huggingface/hub` cache for `depth-anything/Depth-Anything-V2-Small-hf` | [HF model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| `reconstruction_consistency`, `met3r_consistency` | `DA3-GIANT/`, `huggingface/hub` (`naver/MASt3R_*`), `torch/hub` (DINO) | [HF: depth-anything/DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT), [HF: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric), `torch.hub.load("facebookresearch/dino:main", "dino_deitsmall16")` |
| `semantic_alignment`, `background_consistency`, `memory_revisit_consistency` | `huggingface/hub` (`openai/clip-vit-base-patch16`, `facebook/dinov2-base`) | [CLIP](https://huggingface.co/openai/clip-vit-base-patch16), [DINOv2](https://huggingface.co/facebook/dinov2-base) |
| `hpsv3_norm` | `HPSv3/HPSv3.safetensors`, `HPSv3/HPSv3_7B_local.yaml`, `Qwen2-VL-7B-Instruct/` | [HF: MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3), [config template](https://raw.githubusercontent.com/MizzenAI/HPSv3/main/hpsv3/config/HPSv3_7B.yaml), [HF: Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct); also `pip install hpsv3` |
| `segment_continuity` | `TransNetV2/transnetv2-pytorch-weights.pth` + `thirdparty/TransNetV2/inference-pytorch/` | Clone [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2), `git lfs pull`, run `inference-pytorch/convert_weights.py` |
| `jedi` (video suite) | `JEDi/vith16.pth.tar`, `JEDi/ssv2-probe.pth.tar` | [Meta JEPA weights](https://dl.fbaipublicfiles.com/jepa/vith16/) |

> **Note:** `camera_error` defaults to the MegaSAM backend in `config/benchmark.yaml`. MegaSAM requires a separate conda env and upstream checkout; set `MEGASAM_PYTHON` or override `metric_runtime.camera_error` when not using the bundled DROID backend.

### Example: download weights for an adapter

```bash
# Hugging Face CLI (generic pattern)
huggingface-cli download org/model-name --local-dir ckpt/model-name

# ABot-World example
huggingface-cli download acvlab/ABot-World-0-5B-LF --local-dir ckpt/ABot-World-0-5B-LF
```

Then set in `config/models/abot_world.yaml`:

```yaml
model:
  repo_root: thirdparty/ABot-World
  checkpoint_dir: ckpt/ABot-World-0-5B-LF
```

### Checkpoint ↔ config quick reference

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
| `open_dreamer.yaml` | `thirdparty/open-dreamer` | `open-dreamer/<step>/` (Orbax: `tokenizer/`, `dynamics_ema/`) |

Model weights retain their **original licenses**. WorldArena code is [MIT](LICENSE).

---

## Dataset Preparation

### Expected on-disk layout

Benchmark inventory assumes taxonomy-oriented folder trees. Minimal layout for the four formal image/video suites:

```text
data/image/
├── static/
│   └── <style>/<environment>/<scene>/<sample_id>.{jpg,png}
└── dynamic/
    └── <style>/<motion_category>/...
        ├── <sample_id>.{jpg,png}
        ├── <sample_id>_mask.png          # optional
        └── <sample_id>.json              # annotation (prompt, camera_path, …)

data/video/
├── static/
│   └── <style>/<environment>/<scene>/<duration_bucket>/<clip>.mp4
└── dynamic/
    └── …                                 # + annotations / masks when available
```

Annotation JSON (when present) supplies **`prompt_current`**, **`prompt_target`**, optional **`camera_path`**, and **`prompt_sequence`**. WorldArena normalizes these into the benchmark manifest via `worldarena.benchmark.annotations`.

### Build inventory and manifest

Update roots in `config/benchmark.yaml`:

```yaml
paths:
  image_root: data/image    # or absolute path
  video_root: data/video
  inventory_dir: artifacts/benchmark/inventory
  manifest_dir: artifacts/benchmark/manifest
  predictions_dir: artifacts/predictions
  reports_dir: artifacts/benchmark/reports
  cache_dir: .cache/worldarena/benchmark
```

Then run:

```bash
conda activate worldarena

worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest  --config config/benchmark.yaml
```

Outputs:

| Artifact | Location |
|----------|----------|
| Inventory tables | `artifacts/benchmark/inventory/` |
| Benchmark manifest | `artifacts/benchmark/manifest/worldarena_manifest.jsonl` |

For dataset **statistics panels** (docs site / analytics), additionally configure [`config/datasets.yaml`](config/datasets.yaml) and run:

```bash
worldarena build-docs --config config/datasets.yaml
```

---

## File Formats & Contracts

### Benchmark manifest (`*.jsonl`)

One UTF-8 JSON object per line (`BenchmarkSample`). Required fields for generation / evaluation:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Stable identifier, e.g. `image_static_photorealistic_indoor_living_spaces_0001` |
| `suite` | string | `image_static` \| `image_dynamic` \| `video_static` \| `video_dynamic` |
| `path` | string | Absolute or resolvable path to source asset |
| `reference_path` | string | Conditioning reference (usually same as `path`) |
| `prediction_stem` | string | Filename stem for predictions (defaults to `sample_id`) |
| `prompt_current` | string | Scene description at conditioning time |
| `prompt_target` | string | Target prompt for generation |
| `camera_path` | string[] | Symbolic camera / motion tokens (static suites) |
| `conditioning_strategy` | string | `reference_image`, `first_frame`, `reference_video`, … |
| `mask_path` | string \| null | Optional mask for dynamic image metrics |

Example (truncated):

```json
{
  "sample_id": "image_static_photorealistic_outdoor_city_0042",
  "suite": "image_static",
  "path": "data/image/static/photorealistic/outdoor/city/0042.jpg",
  "prediction_stem": "image_static_photorealistic_outdoor_city_0042",
  "prompt_current": "A wide street in a modern city at dusk.",
  "prompt_target": "A wide street in a modern city at dusk.",
  "camera_path": ["forward", "camera_left", "forward"],
  "conditioning_strategy": "reference_image"
}
```

### Prediction outputs

WorldArena resolves predictions in this order:

1. `{predictions_root}/{suite}/{prediction_stem}{ext}`
2. `{predictions_root}/{prediction_stem}{ext}`

Supported video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. Default adapter output: **`.mp4`**.

**Flat layout (common for external submissions):**

```text
predictions/
├── image_static/
│   └── image_static_photorealistic_outdoor_city_0042.mp4
├── image_dynamic/
├── video_static/
└── video_dynamic/
```

**Generation run layout** ( produced by `worldarena generate` ):

```text
artifacts/predictions/<run-name>/
├── image_static_<sample_id>.mp4
├── generation_manifest.json
├── generation_records.jsonl
├── prompt_map.json
└── _conditioning/                 # cached conditioning frames (internal)
```

Sharded runs append `.shard<N>` suffixes to manifest / record files.

### Evaluation reports

`worldarena evaluate` writes to `artifacts/benchmark/reports/<model-name>/` (or `--output-dir`):

| File | Contents |
|------|----------|
| `run_manifest.json` | Run metadata, suite list, missing predictions |
| `per_sample_scores.jsonl` | One JSON object per sample with all metric outputs |
| `per_metric_summary.json` | Suite-level headline metric aggregates |
| `per_dimension_summary.json` | Dimension-level aggregates |
| `leaderboard.csv` | Flat comparison table |
| `not_applicable_summary.json` | Metrics skipped (missing supervision) |
| `diagnostic_metric_summary.json` | Non-headline diagnostic metrics |

Each metric output carries:

| Field | Meaning |
|-------|---------|
| `raw` | Metric-native value |
| `normalized` | Score in `[0, 1]` (higher is better unless documented otherwise) |
| `eligibility_status` | `official`, `diagnostic`, or `not_applicable` |
| `error` | Runtime failure message when `normalized` is null |

### Model config YAML (`config/models/*.yaml`)

Minimal local-repo template:

```yaml
model:
  name: my-world-model
  family: my_world_model
  repo_root: thirdparty/MyModel     # relative to repo root
  checkpoint_dir: ckpt/my-model     # under WORLDARENA_CHECKPOINT_ROOT
  python_bin: python                # or /path/to/dedicated/env/bin/python
  entrypoint: infer.py
  output_ext: .mp4
  supported_suites:
    - image_static
    - video_static
  prompt_mode: prompt_target          # prompt_current | prompt_target
  reference_mode: first_frame         # first_frame | image | …
  generation:
    num_frames: 81
    fps: 24
  env:
    WORLDARENA_CHECKPOINT_ROOT: ckpt
    HF_HOME: ckpt/huggingface
```

HTTP API template: [`config/models/template_video_api.yaml.example`](config/models/template_video_api.yaml.example).

---

## Usage

### End-to-end benchmark (generate + evaluate)

```bash
conda activate worldarena
source .env   # if used

# 1) Build manifest (once per dataset snapshot)
worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest  --config config/benchmark.yaml

# 2) Generate with a registered adapter
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --output-dir artifacts/predictions/lingbot-world-demo \
  --suites video_static \
  --limit 4

# 3) Evaluate predictions
worldarena evaluate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --predictions-root artifacts/predictions/lingbot-world-demo \
  --suites video_static

# One-shot alternative
worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --suites video_static \
  --limit 4
```

### Evaluate externally generated videos

If you already produced videos in the expected layout, skip `generate`:

```bash
worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root /path/to/your/predictions \
  --model-name my_custom_model
```

### Sharded multi-GPU generation

```bash
# Shard 0 of 8
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --output-dir artifacts/predictions/lingbot-world-sharded \
  --num-shards 8 \
  --shard-index 0
```

For CamI2V / RealCam-I2V, see [`scripts/launch_camera_i2v_8gpu.sh`](scripts/launch_camera_i2v_8gpu.sh). For sharded evaluation, see [`scripts/run_eval_shards.py`](scripts/run_eval_shards.py) and [`scripts/README.md`](scripts/README.md).

### CLI reference

| Command | Description |
|---------|-------------|
| `worldarena build-inventory` | Scan benchmark media → inventory artifacts |
| `worldarena build-manifest` | Inventory → `worldarena_manifest.jsonl` |
| `worldarena generate` | Run a model adapter over manifest samples |
| `worldarena evaluate` | Score predictions against the manifest |
| `worldarena run-model` | `generate` + `evaluate` in one step |
| `worldarena build-docs` | Refresh dataset statistics (`config/datasets.yaml`) |
| `worldarena build-site-content` | Export docs into the static site bundle |
| `worldarena build-physics-manifest` | Simulator-backed physics track |
| `worldarena clean` | Remove caches / generated outputs |

---

## Integrating Your Model

**Steps:**

1. Place upstream code under `thirdparty/<Name>/` (or set `repo_root` elsewhere).
2. Download weights to `ckpt/<name>/`.
3. Add `config/models/<name>.yaml` following the schema above.
4. Implement or extend an adapter in `worldarena/models/adapters/` and register it in `worldarena/models/registry.py`.
5. Smoke test: `worldarena generate --limit 1 …` then `worldarena evaluate …`.

For remote video APIs, start from [`config/models/template_video_api.yaml.example`](config/models/template_video_api.yaml.example). Adapter contracts: [`docs/en/architecture.md`](docs/en/architecture.md).

### Example integrations

<details>
<summary><b>LingBot-World</b></summary>

```bash
git submodule update --init thirdparty/lingbot-world
# prepare env per upstream README / requirements

worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --limit 1
```

</details>

<details>
<summary><b>LingBot-Video (TI2V)</b></summary>

```bash
git submodule update --init thirdparty/lingbot-video
# prepare env per upstream README (e.g. pip install -r thirdparty/lingbot-video/requirements.txt)

worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_video_dense.yaml
```

</details>

<details>
<summary><b>ABot-World</b></summary>

```bash
git submodule update --init thirdparty/ABot-World
# prepare env per upstream README
huggingface-cli download acvlab/ABot-World-0-5B-LF --local-dir ckpt/ABot-World-0-5B-LF

worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/abot_world.yaml
```

</details>

<details>
<summary><b>Matrix-Game 3.5</b></summary>

```bash
bash scripts/setup_matrix_game_35.sh

worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/matrix_game_35_first.yaml
```

</details>

<details>
<summary><b>Open Dreamer</b></summary>

```bash
bash scripts/setup_open_dreamer.sh   # submodule + uv venv (JAX/CUDA 12) + Orbax weights

worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/open_dreamer.yaml
```

Open Dreamer takes **no text prompt**: the only control signal is one Minecraft/VPT
action per frame, so `open_dreamer_actions.py` translates each `camera_path` segment
into keyboard/mouse events (`push_in` → `w`, `pan_right` → mouse `dx`, …) laid out
over the same frame boundaries the camera metrics score against. `roll_cw` /
`roll_ccw` segments are rejected because the VPT action space has no roll axis, and
the model is trained on 368×640 Minecraft footage — expect out-of-domain behaviour
on natural video.

</details>

---

## Supported Models

| Model | Config | Formal suites |
|-------|--------|---------------|
| CamI2V | `cami2v.yaml` | image_static, image_dynamic |
| RealCam-I2V | `realcam_i2v.yaml` | image_static, image_dynamic |
| ABot-World 0.5B | `abot_world.yaml` | image_static, image_dynamic |
| AlayaWorld 15B | `alayaworld.yaml` | all four (first-frame rollout) |
| LingBot-World | `lingbot_world.yaml` | all four |
| LingBot-World (fast) | `lingbot_world_fast.yaml` | all four |
| LingBot-Video Dense | `lingbot_video_dense.yaml` | all four (TI2V) |
| LingBot-Video MoE | `lingbot_video_moe.yaml` | all four (TI2V) |
| HY-WorldPlay | `hy_worldplay.yaml` | image_static, video_static, video_dynamic |
| WorldCam | `worldcam.yaml` | image_static |
| FlashWorld | `flashworld.yaml` | video_static |
| NeoVerse | `neoverse.yaml` | video_static, video_dynamic |
| GEN3C | `gen3c.yaml` | image_static |
| Lyra-1 / Lyra-2 | `lyra1.yaml`, `lyra2.yaml` | image / video static |
| InSpatio-World | `inspatio_world.yaml` | image_static, video_static |
| Infinite-World | `infinite_world.yaml` | image_static, video_dynamic |
| WonderWorld / WonderJourney | `wonderworld.yaml`, `wonderjourney.yaml` | image / video static |
| HunyuanWorld | `hunyuanworld.yaml` | image_static |
| LucidDreamer | `luciddreamer.yaml` | image_static, video_static |
| YUME | `yume.yaml` | image_static, video_static |
| WorldFM | `worldfm.yaml` | video_static |
| MatrixGame 1/2/3 | `matrix_game*.yaml` | image_dynamic, video_dynamic |
| Matrix-Game 3.5 | `matrix_game_35_*.yaml` | image_static, image_dynamic |
| Hunyuan GameCraft | `hunyuan_gamecraft.yaml` | image_dynamic, video_dynamic |
| Cosmos-Predict 2.5 | `cosmos_predict25_*.yaml` | image_static, video_static |
| Open Dreamer | `open_dreamer.yaml` | all four (VPT action rollout) |

---

## Documentation & Site

| Resource | Path / command |
|----------|----------------|
| English docs source | `docs/en/` |
| Chinese docs source | `docs/zh/` |
| Build docs (Astro + Starlight) | `npm run docs:build` |
| Preview docs locally | `npm run docs:dev` → `http://localhost:4321/doc/` |
| Leaderboard & arena site | `npm run site` → `http://localhost:4173/` |
| Editable leaderboard results | `site/leaderboard-results.json` |
| Site OAuth setup | [`site/AUTH.md`](site/AUTH.md), [`site/.env.example`](site/.env.example) |

The leaderboard reads only `site/leaderboard-results.json` at runtime. Edit `overall` or any value under `scores` directly; the running site reloads the file after it is saved. To replace manual values with a fresh import from benchmark reports, run `npm run leaderboard:import-reports`.

Full contributor workflow: [`docs/en/development.md`](docs/en/development.md).

---

## Repository Layout

```text
WorldArena/
├── worldarena/                 # Python package
│   ├── benchmark/              # inventory, manifest, metrics, runner, reports
│   │   ├── metrics/            # 30+ metric implementations
│   │   └── official_backends/  # in-tree adapters over upstream checkouts
│   ├── models/adapters/        # per-model generation adapters
│   ├── generation/             # prediction runner
│   ├── datasets/               # dataset scanners & summarizers
│   └── cli.py                  # `worldarena` entry point
├── config/
│   ├── benchmark.yaml          # suites, metrics, paths
│   ├── datasets.yaml           # analytics scanners
│   └── models/                 # adapter YAML configs
├── docs/                       # Markdown, assets, and Astro / Starlight builder
├── site/                       # static leaderboard & arena UI
├── scripts/                    # optional helpers (see scripts/README.md)
├── tests/                      # adapter & metric unit tests
├── thirdparty/                 # git submodules (upstream models)
├── artifacts/                  # generated outputs (gitignored)
├── data/                       # benchmark media (gitignored, user-provided)
└── ckpt/                       # weights & HF cache (gitignored)
```

---

## Citation

If you use WorldArena in your research, please cite:

```bibtex
@article{worldarena2026,
  title   = {WorldArena: A Unified Benchmark for World Model Evaluation},
  author  = {OpenEnvision},
  year    = {2026},
  url     = {https://github.com/OpenEnvision/WorldArena}
}
```

---

## License

- **Code in this repository:** [MIT License](LICENSE)
- **Model weights:** subject to each upstream project's license
- **Benchmark media:** subject to the dataset license under which you obtained the assets

---

<div align="center">

**WorldArena** · [OpenEnvision](https://openenvision.github.io/index.html) · [GitHub](https://github.com/OpenEnvision/WorldArena) · [Documentation](docs/en/index.md)

</div>
