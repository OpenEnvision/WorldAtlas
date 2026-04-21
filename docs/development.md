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

If an adapter needs a separate interpreter, bootstrap `.runtime312` and keep adapter-specific packages there:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  opencv-python==4.10.0.84
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
