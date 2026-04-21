# WorldArena

WorldArena is an extensible benchmark workspace for world model evaluation, dataset inspection, and documentation generation. The repository is designed to stay stable at the benchmark-contract layer while model adapters, third-party repos, checkpoints, and metric backends evolve independently.

This repo does not vendor upstream model repositories or redistribute model weights. Instead, it provides:

- the benchmark inventory, manifest, and reporting contract
- official metric execution for the formal suites
- adapter configs for selected world models under `config/models/`
- a documented runtime layout for `cache/` and `third_party/`
- a runtime self-check command for open-source users

## Install Profiles

Core benchmark and dataset analytics:

```bash
uv sync --python 3.10
```

Official metric backends:

```bash
uv sync --python 3.10 --extra eval --extra quality
```

Heavy video metrics:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
./.venv/bin/python -m spacy download en_core_web_sm
```

Heavy video metrics also need a compatible metric-source checkout wired either through `WORLDARENA_METRIC_SOURCE_ROOT` or under `third_party/`.

`video-metrics` includes the heavier official dependencies such as `cvxpy`, `iopath`, `pycocotools`, `causal-conv1d`, and `mamba-ssm`. If `mamba-ssm` wheel resolution fails on your machine, install the exact `causal-conv1d` / `mamba-ssm` pair recommended by the compatible metric-source checkout and rerun `worldarena check-env`.

Integrated world-model adapters:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

## Prepare Runtime Layout

Create the open-source-friendly local layout:

```bash
uv run worldarena prepare-layout
```

Reuse an existing checkpoint cache and link local model repos into the expected locations:

```bash
uv run worldarena prepare-layout \
  --shared-hf-cache /path/to/shared/hf-cache \
  --repo-link lingbot-world=/path/to/lingbot-world \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0 \
  --repo-link metric-source=/path/to/compatible-metric-checkout
```

Default contract:

```text
cache/
  hfd/
third_party/
  lingbot-world/
  Matrix-Game-3.0/
```

Tracked guidance lives in `cache/README.md` and `third_party/README.md`. Both directories are git-ignored except for those READMEs so that local clones, symlinks, and large checkpoints stay out of version control.

Inspect the local setup after wiring it:

```bash
uv run worldarena check-env
```

`check-env` is a static readiness probe. It reports layout, dependency, metric-source, and GPU status without instantiating heavy metric classes.

## Benchmark Workflow

WorldArena keeps four formal suites as the public benchmark core:

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

Official headline reporting is a per-suite metric vector, not a suite-level scalar and not a benchmark-wide total score.

Build inventory and manifest:

```bash
uv run worldarena build-inventory --config config/benchmark.yaml
uv run worldarena build-manifest --config config/benchmark.yaml
```

Generate outputs with an adapter config:

```bash
uv run worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Evaluate an existing predictions folder:

```bash
uv run worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root /path/to/predictions
```

Or run the one-shot path:

```bash
uv run worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Expected prediction layout:

```text
predictions/
  image_static/<sample_id>.png
  image_dynamic/<sample_id>.png
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

Run artifacts include:

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `diagnostic_metric_summary.json` when any metric is non-official for eligibility reasons
- `domain_breakdown.json`
- `leaderboard.csv`

## Official Reporting Contract

Current official metric dimensions are:

- `content_alignment`: `prompt_alignment`
- `style_consistency`: `style_consistency`
- `perceptual_quality`: `perceptual_quality`
- `camera_control`: `camera_error`
- `object_control`: `object_detection`
- `geometry_consistency`: `reprojection_error`
- `photometric_consistency`: `optical_flow_aepe`
- `motion_control`: `motion_accuracy`
- `motion_dynamics`: `motion_magnitude`, `motion_smoothness`

Current suite assignments are:

- `image_static`: `style_consistency`, `perceptual_quality`
- `image_dynamic`: `style_consistency`, `perceptual_quality`
- `video_static`: `style_consistency`, `perceptual_quality`, `camera_error`, `object_detection`, `reprojection_error`, `optical_flow_aepe`
- `video_dynamic`: `style_consistency`, `perceptual_quality`, `motion_accuracy`, `motion_magnitude`, `motion_smoothness`

`prompt_alignment` is still computed, but on self-conditioned samples with template-derived prompts it is reported as `weak_supervision` and routed to diagnostics instead of the official aggregate.

`object_detection` is also routed to `weak_supervision` when object prompts have to be derived from caption text instead of explicit object annotations.

Non-formal assets stay outside official headline reporting and are routed into separate `auxiliary` or `experimental` reporting.

## Adapter Runtime Contract

Bundled model configs now use project-relative defaults so the repository can be published without private absolute paths:

- `config/models/lingbot_world.yaml`
  - `repo_root: third_party/lingbot-world`
  - `checkpoint_dir: cache/hfd/robbyant--lingbot-world-base-cam`
- `config/models/matrix_game.yaml`
  - `repo_root: third_party/Matrix-Game-3.0`
  - `checkpoint_dir: cache/hfd/Matrix-Game-3.0`

If the upstream model code needs a separate CUDA-enabled interpreter, the configs still support `python_bin: auto`, which prefers `.runtime312/bin/python` when present.

Example runtime bootstrap for adapters that need a 3.12 side environment:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  opencv-python==4.10.0.84
```

Optional model-side environment overrides live under `model.env` in the YAML configs. Use that block for `CUDA_VISIBLE_DEVICES`, `HF_HOME`, or other adapter-specific runtime settings.

## Dataset Analytics And Docs

Refresh the dataset analytics pages:

```bash
uv run worldarena build-docs --config config/datasets.yaml
```

Preview the documentation site locally:

```bash
uv run mkdocs serve
```

Additional documentation:

- `docs/benchmark.md`: benchmark scope, metric policy, reporting contract
- `docs/environment.md`: install profiles, self-check flow, runtime layering
- `docs/third_party.md`: third-party repos, cache layout, and checkpoint integration
- `docs/development.md`: maintainer workflow, validation commands, release hygiene
