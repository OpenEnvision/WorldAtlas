# Development

All commands below assume the project's conda environment is already activated.

## Dataset Analytics

```bash
worldarena build-docs --config config/datasets.yaml
```

This command scans the configured dataset roots, refreshes `artifacts/dataset_stats/`, and rewrites `docs/generated/`.

## Documentation Site

Install the MkDocs toolchain from the dedicated docs requirements file:

```bash
python -m pip install -r requirements/docs.txt
```

Preview the bilingual documentation site locally:

```bash
mkdocs serve
```

Build the site with strict validation:

```bash
mkdocs build --strict
```

## Benchmark

```bash
worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest --config config/benchmark.yaml
worldarena evaluate --config config/benchmark.yaml --predictions-root /path/to/preds
```

WorldArena's official benchmark surface is intentionally narrow:

- official suites: `image_static`, `image_dynamic`, `video_static`, `video_dynamic`
- official headline output: per-metric vectors, not a suite score and not a global total score
- metrics without strong supervision are marked `not_applicable`, not mixed into official or diagnostic reporting

Simulator-backed physics workflows live on a separate config:

```bash
worldarena build-physics-manifest --physics-pipeline-root /path/to/physics-pipeline
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --physics-pipeline-root /path/to/physics-pipeline
```

## Integrated Model Adapters

Bundled adapters assume open-source-friendly relative paths in `config/models/*.yaml`. If an upstream repo needs a different layout, prefer overriding the YAML locally or linking the repo into `thirdparty/` instead of hard-coding private absolute paths back into versioned config.

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

## Preview And Validation

```bash
mkdocs build --strict
pytest
ruff check .
```

For a lighter pre-commit validation pass, at minimum run:

```bash
pytest -q tests/test_model_config.py tests/test_runtime_layout.py tests/test_environment_check.py
```

If you touched heavy runtime wrappers or simulator-backed evaluation, also run the targeted suites that cover those paths, such as `tests/test_long_sequence_backend_runtime.py`, `tests/test_long_sequence_runtime.py`, `tests/test_long_sequence_native.py`, and `tests/test_physics_simulator_derived_metrics.py`.

## Cleanup

```bash
worldarena clean
```

Add `--generated` if you also want to remove `docs/generated/` and `artifacts/dataset_stats/`.

## Open-Source Release Hygiene

Before publishing changes, check the following:

- `config/models/*.yaml` should use project-relative defaults, not private absolute paths.
- `cache/` and `thirdparty/` should only keep their tracked README files under version control.
- documentation should explain any new checkpoint, repo, or license assumptions for added adapters.
- generated benchmark reports and dataset artifacts should stay out of git unless they are intentionally checked in as fixtures.
