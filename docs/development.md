# Development

## Environment Profiles

Core benchmark and dataset analytics:

```bash
uv sync --python 3.10
```

Official metric backends:

```bash
uv sync --python 3.10 --extra eval --extra quality
```

Temporal calibration metrics:

```bash
uv sync --python 3.10 --extra temporal-calibration
```

Heavy video metrics:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
./.venv/bin/python -m spacy download en_core_web_sm
```

If `mamba-ssm` is not available as a wheel for the target machine, install the exact `causal-conv1d` / `mamba-ssm` pair documented by the compatible metric-source checkout before running the heavy video metrics. The same environment should also carry `iopath` for the `sam2` path used by `motion_accuracy`.

Integrated world-model adapters:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

## Prepare Local Layout

Before wiring in upstream repos or large checkpoints, create the local layout expected by the bundled configs:

```bash
uv run worldarena prepare-layout
```

If you already have machine-local caches or external clones:

```bash
uv run worldarena prepare-layout \
  --shared-hf-cache /path/to/shared/hf-cache \
  --repo-link lingbot-world=/path/to/lingbot-world \
  --repo-link Lyra-1=/path/to/Lyra-1 \
  --repo-link Lyra-2=/path/to/Lyra-2 \
  --repo-link HunyuanWorld-1.0=/path/to/HunyuanWorld-1.0 \
  --repo-link worldfm=/path/to/worldfm \
  --repo-link WorldCam=/path/to/WorldCam \
  --repo-link Matrix-Game-1=/path/to/Matrix-Game-1 \
  --repo-link Matrix-Game-2=/path/to/Matrix-Game-2 \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0
```

This keeps the repository publishable while letting your local checkout reuse heavyweight assets.

Validate the environment state before running larger experiments:

```bash
uv run worldarena check-env
```

## Dataset Analytics

```bash
uv run worldarena build-docs --config config/datasets.yaml
```

This command scans the configured dataset roots, refreshes `artifacts/dataset_stats/`, and rewrites `docs/generated/`.

## Benchmark

```bash
uv run worldarena build-inventory --config config/benchmark.yaml
uv run worldarena build-manifest --config config/benchmark.yaml
uv run worldarena evaluate --config config/benchmark.yaml --predictions-root /path/to/preds
```

WorldArena's official benchmark surface is intentionally narrow:

- official suites: `image_static`, `image_dynamic`, `video_static`, `video_dynamic`
- official headline output: per-metric vectors, not a suite score and not a global total score
- diagnostics such as weak-supervision `prompt_alignment`: visible, but not mixed into official reporting

## Integrated Model Adapters

Bundled adapters assume open-source-friendly relative paths in `config/models/*.yaml`. If an upstream repo needs a different layout, prefer overriding the YAML locally or linking the repo into `third_party/` instead of hard-coding private absolute paths back into versioned config.

All models, including future API-backed ones, should keep the same YAML surface:

- `backend_type`
- `task_families`
- `artifact_types`
- `control_signals`
- `roles`

Use `config/models/template_video_api.yaml.example` as the starting point for API models instead of introducing a second config format.

The current API adapter path is the generic benchmark path, not a service-specific side framework:

- `backend_type: http_api`
- one request builder for both `t2v` and `i2v`
- provider-specific submission and polling under `model.api`
- no special-case benchmark command path for remote models

`config/models/kling_api.yaml.example` is the concrete reference config. The API pipeline regression tests run against a local fake polling server, so they validate the benchmark contract without needing real credentials.

If an adapter needs a separate interpreter, bootstrap `.runtime312` and keep adapter-specific packages there:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  modelscope sentencepiece protobuf pynvml opencv-python==4.10.0.84
```

## Preview And Validation

```bash
uv run mkdocs build --strict
uv run pytest
uv run ruff check .
```

For a lighter pre-commit validation pass, at minimum run:

```bash
uv run pytest -q tests/test_model_config.py tests/test_runtime_layout.py tests/test_environment_check.py
```

## Cleanup

```bash
uv run worldarena clean
```

Add `--generated` if you also want to remove `docs/generated/` and `artifacts/dataset_stats/`.

## Open-Source Release Hygiene

Before publishing changes, check the following:

- `config/models/*.yaml` should use project-relative defaults, not private absolute paths.
- `cache/` and `third_party/` should only keep their tracked README files under version control.
- documentation should explain any new checkpoint, repo, or license assumptions for added adapters.
- generated benchmark reports and dataset artifacts should stay out of git unless they are intentionally checked in as fixtures.
