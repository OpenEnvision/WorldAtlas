# Environment

WorldArena has two runtime layers:

- the repo-managed benchmark environment, created with `uv`
- an optional side runtime such as `.runtime312/` for adapters or heavy metric stacks that are easier to isolate

The benchmark code itself stays in one repository, but not every metric or model backend should share one Python environment.

## Install Profiles

Core benchmark and dataset analytics:

```bash
uv sync --python 3.10
```

Reference image/video metrics used by the default benchmark path:

```bash
uv sync --python 3.10 --extra eval --extra quality
```

Temporal calibration metrics:

```bash
uv sync --python 3.10 --extra temporal-calibration
```

This profile covers the upstream `Pulse-of-Motion` runtime pieces that are not already in the base environment, including `pytorch_lightning`, `einops`, `omegaconf`, and `huggingface_hub`.

Heavy video metrics that reuse the official algorithms:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
./.venv/bin/python -m spacy download en_core_web_sm
```

That profile is also where the official dynamic stack gets `iopath`, `pycocotools`, `causal-conv1d`, and `mamba-ssm`. If `mamba-ssm` cannot be resolved as a wheel on your machine, install the exact `causal-conv1d` / `mamba-ssm` pair recommended by the compatible metric-source checkout and rerun `worldarena check-env`.

`Pulse-of-Motion` lives alongside that stack. It is not vendored into the repo; the benchmark looks for it under `third_party/Pulse-of-Motion` unless `WORLDARENA_PULSE_OF_MOTION_ROOT` is set.

Its upstream checkpoint is normally downloaded on first use. `worldarena check-env` therefore treats the checkout as ready when the source files are present and either the checkpoint is already cached or `huggingface_hub` is available for auto-download.

Bundled model adapters on top of the benchmark stack:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

## Runtime Layout

Create the publishable local layout first:

```bash
uv run worldarena prepare-layout
```

If you already have a shared cache or local upstream clones:

```bash
uv run worldarena prepare-layout \
  --shared-hf-cache /path/to/shared/hf-cache \
  --repo-link lingbot-world=/path/to/lingbot-world \
  --repo-link Infinite-World=/path/to/Infinite-World \
  --repo-link Lyra-1=/path/to/Lyra-1 \
  --repo-link Lyra-2=/path/to/Lyra-2 \
  --repo-link WorldCam=/path/to/WorldCam \
  --repo-link HunyuanWorld-1.0=/path/to/HunyuanWorld-1.0 \
  --repo-link worldfm=/path/to/worldfm \
  --repo-link Matrix-Game-1=/path/to/Matrix-Game-1 \
  --repo-link Matrix-Game-2=/path/to/Matrix-Game-2 \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0 \
  --repo-link metric-source=/path/to/compatible-metric-checkout \
  --repo-link Pulse-of-Motion=/path/to/Pulse-of-Motion
```

If more than one compatible metric-source checkout exists, pin the one you want:

```bash
export WORLDARENA_METRIC_SOURCE_ROOT=$PWD/third_party/metric-source
export WORLDARENA_PULSE_OF_MOTION_ROOT=$PWD/third_party/Pulse-of-Motion
```

## Self Check

After installation, inspect the local environment:

```bash
uv run worldarena check-env
```

The report is JSON and covers:

- current Python executable and version
- cache and `third_party/` layout
- discovered runtime interpreters such as `.runtime312/bin/python`
- dependency profiles for core, reference metrics, heavy video metrics, and model adapters
- metric-source resolution status
- Pulse-of-Motion resolution status
- per-metric backend readiness grouped by evaluation dimension
- visible GPUs when `nvidia-smi` is available

`worldarena check-env` is intentionally a static probe. It validates source layout, importable packages, declared metric entries, and visible GPUs without instantiating heavy metric classes or loading model weights.

It also does not validate remote API credentials. For API-backed models, `check-env` verifies the local config surface and dependency state, but not whether a given service key is valid.

## Recommended Flow

For a fresh machine:

1. `uv sync --python 3.10 --extra eval --extra quality`
2. `uv run worldarena prepare-layout`
3. `uv run worldarena check-env`
4. add `--extra temporal-calibration` if you want the `Pulse-of-Motion` metrics
5. add `--extra video-metrics` only if you need the heavy official video metrics
6. add `--group world_models` only if you need bundled model adapters

## Optional Side Runtime

Some adapters and source checkouts are cleaner in an isolated interpreter:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  opencv-python==4.10.0.84
```

Keep benchmark orchestration in the repo environment and reserve `.runtime312/` for source-specific packages when necessary.

`Infinite-World`, `Lyra-1`, `Lyra-2`, `WorldFM`, and `HunyuanWorld` usually need their own `.runtime310/` instead of `.runtime312/`, because these upstream releases target Python 3.10 and dedicated CUDA-compatible stacks.

## API Models

API-backed models use the same YAML schema as local adapters:

- `backend_type: http_api`
- `repo_root: null`
- provider and polling contract under `model.api`
- generation knobs under `model.generation`

Use `config/models/template_video_api.yaml.example` as the baseline and `config/models/kling_api.yaml.example` as the concrete example. The current adapter supports both `t2v` and `i2v` through the same request builder, so the benchmark-side workflow stays unchanged.

## Common Gaps

- `camera_error` usually fails when `cvxpy` or compiled `droid/lietorch` pieces are missing.
- `object_detection` and `motion_accuracy` need `pycocotools`; `motion_accuracy` also needs the `sam2` import chain to resolve cleanly, including `iopath`, plus both `spacy` and the `en_core_web_sm` model.
- `motion_smoothness` depends on the `VFIMamba` source tree and its `timm/einops/causal-conv1d/mamba-ssm` stack.
- `metric_source` not found means the source checkout was not pinned or not linked under `third_party/`.
- `pulse_of_motion` not ready means `predict.py` or `config_fps.yaml` is missing, or the checkpoint is absent and `huggingface_hub` is unavailable for upstream auto-download.

## Heavy Dynamic Runtime

`motion_accuracy` has an additional runtime policy layer under `benchmark.metric_runtime.motion_accuracy` in `config/benchmark.yaml`.

- `sam2_offload_video_to_cpu: true` keeps decoded video frames on CPU memory.
- `sam2_offload_state_to_cpu: true` moves SAM2 tracking state off GPU memory between steps.
- `sam2_device`, `flow_device`, and `grounding_dino_device` can be set explicitly on multi-GPU hosts when you want SAM2 and the follow-up optical-flow stage on different devices.

The default config enables the two SAM2 offload flags because they materially reduce peak memory during official dynamic evaluation.
