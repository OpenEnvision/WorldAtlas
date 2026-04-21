# WorldArena

WorldArena is an extensible benchmark workspace for world model evaluation, dataset inspection, and documentation generation.

## Quickstart

```bash
uv sync
uv run worldarena build-docs --config config/datasets.yaml
uv run mkdocs serve
```

The dataset analytics pipeline scans the image and video roots declared in `config/datasets.yaml`, writes structured artifacts to `artifacts/dataset_stats/`, and regenerates the MkDocs pages in `docs/generated/`.

Use `uv run worldarena clean` to remove transient caches and local build outputs. Add `--generated` if you also want to wipe generated docs pages and dataset artifacts.

## Benchmark Workflow

WorldArena keeps four formal suites as the public core, matching a simpler WorldScore-style evaluation surface:

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

Non-formal assets stay outside the official score and are routed into `auxiliary` or `experimental`.

### Build Inventory And Manifest

```bash
uv sync --python 3.10
uv run worldarena build-inventory --config config/benchmark.yaml
uv run worldarena build-manifest --config config/benchmark.yaml
```

This writes:

- `artifacts/benchmark/inventory/worldarena_inventory.json`
- `artifacts/benchmark/inventory/worldarena_inventory.csv`
- `artifacts/benchmark/manifest/worldarena_manifest.jsonl`
- `artifacts/benchmark/manifest/worldarena_suite_summary.csv`

### Generate Model Outputs

Model runtime stays adapter-based: the benchmark contract lives here, while the actual model code can stay in the upstream repo or a sibling workspace.

```bash
uv run worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --limit 8
```

```bash
uv run worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/matrix_game.yaml \
  --suites video_static,video_dynamic \
  --limit 8
```

### Evaluate Predictions

Core lightweight backends work with the default environment. For CLIP/VGG-style model backends, sync the eval extra:

```bash
uv sync --extra eval
```

For the two first integrated world models, install the runtime group as well:

```bash
uv sync --python 3.10 --group world_models --extra eval
```

Then run:

```bash
uv run worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root /path/to/predictions \
  --model-name my_model
```

Expected prediction layout:

```text
/path/to/predictions/
  image_static/<sample_id>.png
  image_dynamic/<sample_id>.png
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

Or use the integrated one-shot path:

```bash
uv run worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml \
  --limit 8
```

The runner writes:

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `domain_breakdown.json`
- `leaderboard.csv`
