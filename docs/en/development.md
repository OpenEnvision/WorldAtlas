# Development

<p class="dimension-lede">This page is the contributor workflow index for WorldAtlas Arena: environment setup pointers, dataset refresh, benchmark commands, documentation build, and validation. For module boundaries and contracts, see <a href="architecture/">Architecture</a>. For the full runtime install path, see <a href="thirdparty/">Runtime Setup</a>.</p>

All commands below assume the project Conda environment (`worldarena`) is activated and the package is installed editable:

```bash
conda activate worldarena
pip install -e .
```

## Repository layout

| Path | Role |
| --- | --- |
| `worldarena/` | Benchmark core: manifests, metrics, reporting, model adapters |
| `config/` | Benchmark, dataset, and model YAML contracts |
| `docs/en/`, `docs/zh/` | Authoritative documentation source (Markdown) |
| `docs/site-nav.yaml` | Sidebar and hidden-page contract for the docs site |
| `docs-site/` | Astro v6 + Starlight builder; output goes to `site/doc/` |
| `site/` | Static site shell, leaderboard, and `/doc/` hosting |
| `thirdparty/` | Linked upstream model and metric runtimes |
| `artifacts/` | Predictions, reports, and generated dataset stats |

## Dataset analytics

Refresh dataset inventory panels and generated analytics artifacts:

```bash
worldarena build-docs --config config/datasets.yaml
```

This scans configured dataset roots, updates `artifacts/dataset_stats/`, and rewrites `docs/generated/`. Re-run after adding, removing, or reorganizing source assets. Cached panels on the analytics pages reflect the last successful run.

## Documentation site

Documentation is built with **Astro + Starlight** (not MkDocs). Source lives in `docs/{en,zh}/`; navigation is driven by `docs/site-nav.yaml`.

**Install docs toolchain** (Node.js 18+):

```bash
cd docs-site
npm install
```

**Local preview** (hot reload on `http://localhost:4321/doc/`):

```bash
npm run dev
```

**Production build** (writes static HTML to `site/doc/`):

```bash
# from repo root
npm run docs:build
```

The build pipeline automatically:

1. Syncs `docs/{en,zh}` into `docs-site/src/content/docs/` (hyphenated slugs, asset path rewrites).
2. Regenerates `docs-site/src/sidebar.generated.mjs`.
3. Links shared assets from `docs/assets/` into the docs public tree.

**Preview with the main site** (serves `/doc/` alongside leaderboard and arena):

```bash
npm run docs:build
npm run site
# default http://localhost:4173/doc/
```

Legacy `/docs/*` URLs redirect to `/doc/*`. When editing metric pages with embedded case media, ensure `docs/assets/metric_examples/` is present; the sync step symlinks it under `/assets/worldarena/assets/metric_examples/`.

See `docs-site/README.md` for route mapping and customization (`docs-site/src/styles/custom.css`).

## Benchmark

Standard formal-suite workflow:

```bash
worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest --config config/benchmark.yaml

# generate predictions with a registered adapter
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml

# evaluate an existing predictions tree
worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root artifacts/predictions/<model-run>

# or one-shot generate + evaluate
worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

WorldAtlas Arena keeps the public benchmark surface intentionally narrow:

- **Official suites:** `image_static`, `image_dynamic`, `video_static`, `video_dynamic`
- **Headline output:** per-metric vectors in `per_metric_summary.json` and dimension rollups in `per_dimension_summary.json` — not a single suite score or global total
- **Eligibility:** metrics without strong supervision are marked `not_applicable` and excluded from official aggregates

Simulator-backed physics uses a separate config and report namespace:

```bash
worldarena build-physics-manifest --physics-pipeline-root /path/to/physics-pipeline
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --physics-pipeline-root /path/to/physics-pipeline
```

See [Benchmark Overview](benchmark/index.md) and [Physics dimension](benchmark/metrics/dimensions/physics.md) for interpretation boundaries.

## Integrated model adapters

Bundled adapters assume open-source-friendly relative paths in `config/models/*.yaml`. If an upstream repo needs a different layout, override the YAML locally or link the repo into `thirdparty/` instead of hard-coding private absolute paths into versioned config.

All models — local repos, Python modules, and HTTP APIs — share one YAML surface:

| Field | Purpose |
| --- | --- |
| `backend_type` | `local_repo`, `python_module`, `sdk_api`, or `http_api` |
| `task_families` | e.g. `video_generation`, `world_model` |
| `artifact_types` | output modality (`video`, `mesh`, …) |
| `control_signals` | conditioning axes used for eligibility |
| `roles` | `candidate`, `oracle`, `metric_backend`, … |

For API models, start from `config/models/template_video_api.yaml.example` rather than introducing a second config format. The generic path is:

- `backend_type: http_api`
- one request builder for both `t2v` and `i2v`
- provider-specific submit/poll logic under `model.api`
- no special-case benchmark CLI branch for remote models

`config/models/kling_api.yaml.example` is the concrete reference. API pipeline tests run against a local fake polling server, so the benchmark contract can be validated without live credentials.

## Preview and validation

**Documentation**

```bash
npm run docs:build
```

**Python**

```bash
pytest
ruff check .
```

Minimum pre-commit pass when touching model configs or runtime layout:

```bash
pytest -q tests/test_model_config.py tests/test_runtime_layout.py tests/test_environment_check.py
```

If you change heavy runtime wrappers or simulator-backed evaluation, also run targeted suites such as:

- `tests/test_long_sequence_backend_runtime.py`
- `tests/test_long_sequence_runtime.py`
- `tests/test_long_sequence_native.py`
- `tests/test_physics_simulator_derived_metrics.py`

After editing docs navigation or Markdown under `docs/`, run `npm run docs:build` and spot-check `/doc/` in the main site.

## Site content export

To refresh the legacy SPA content bundle (`site/js/content*.js`) used by embedded doc views:

```bash
worldarena build-site-content
```

This reads `docs/site-nav.yaml` and exports page bodies for the static site shell. Starlight pages under `/doc/` are built separately via `npm run docs:build`.

## Cleanup

```bash
worldarena clean
```

Add `--generated` to also remove `docs/generated/` and `artifacts/dataset_stats/`.
