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

Heavy video metrics that reuse the official algorithms:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
./.venv/bin/python -m spacy download en_core_web_sm
```

That profile is also where the official dynamic stack gets `iopath`, `pycocotools`, `causal-conv1d`, and `mamba-ssm`. If `mamba-ssm` cannot be resolved as a wheel on your machine, install the exact `causal-conv1d` / `mamba-ssm` pair recommended by the compatible metric-source checkout and rerun `worldarena check-env`.

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
  --repo-link WorldCam=/path/to/WorldCam \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0 \
  --repo-link metric-source=/path/to/compatible-metric-checkout
```

If more than one compatible metric-source checkout exists, pin the one you want:

```bash
export WORLDARENA_METRIC_SOURCE_ROOT=$PWD/third_party/metric-source
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
- per-metric backend readiness grouped by evaluation dimension
- visible GPUs when `nvidia-smi` is available

`worldarena check-env` is intentionally a static probe. It validates source layout, importable packages, declared metric entries, and visible GPUs without instantiating heavy metric classes or loading model weights.

## Recommended Flow

For a fresh machine:

1. `uv sync --python 3.10 --extra eval --extra quality`
2. `uv run worldarena prepare-layout`
3. `uv run worldarena check-env`
4. add `--extra video-metrics` only if you need the heavy official video metrics
5. add `--group world_models` only if you need bundled model adapters

## Optional Side Runtime

Some adapters and source checkouts are cleaner in an isolated interpreter:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  opencv-python==4.10.0.84
```

Keep benchmark orchestration in the repo environment and reserve `.runtime312/` for source-specific packages when necessary.

## Common Gaps

- `camera_error` usually fails when `cvxpy` or compiled `droid/lietorch` pieces are missing.
- `object_detection` and `motion_accuracy` need `pycocotools`; `motion_accuracy` also needs the `sam2` import chain to resolve cleanly, including `iopath`, plus both `spacy` and the `en_core_web_sm` model.
- `motion_smoothness` depends on the `VFIMamba` source tree and its `timm/einops/causal-conv1d/mamba-ssm` stack.
- `metric_source` not found means the source checkout was not pinned or not linked under `third_party/`.

## Heavy Dynamic Runtime

`motion_accuracy` has an additional runtime policy layer under `benchmark.metric_runtime.motion_accuracy` in `config/benchmark.yaml`.

- `sam2_offload_video_to_cpu: true` keeps decoded video frames on CPU memory.
- `sam2_offload_state_to_cpu: true` moves SAM2 tracking state off GPU memory between steps.
- `sam2_device`, `flow_device`, and `grounding_dino_device` can be set explicitly on multi-GPU hosts when you want SAM2 and the follow-up optical-flow stage on different devices.

The default config enables the two SAM2 offload flags because they materially reduce peak memory during official dynamic evaluation.
