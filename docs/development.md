# Development

## Environment

```bash
uv sync --python 3.10
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

The benchmark stack is intentionally narrower than the full `plan.md` taxonomy:

- official suites: `image_static`, `image_dynamic`, `video_static`, `video_dynamic`
- extra panels: `auxiliary`, `experimental`
- default metric families: semantic alignment, style consistency, subjective quality, rollout quality, motion magnitude, motion smoothness

If you want CLIP/VGG-based backends instead of the default lightweight fallbacks, install:

```bash
uv sync --extra eval
```

If you want to run the first integrated world-model backends from this repo:

```bash
uv sync --python 3.10 --group world_models --extra eval
uv run worldarena generate --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
uv run worldarena generate --config config/benchmark.yaml --model-config config/models/matrix_game.yaml
```

## Preview And Validation

```bash
uv run mkdocs serve
uv run mkdocs build --strict
uv run pytest
uv run ruff check .
```

## Cleanup

```bash
uv run worldarena clean
```

Add `--generated` if you also want to remove `docs/generated/` and `artifacts/dataset_stats/`.

## Extension Points

- Add new dataset families by registering another scanner in `worldarena/datasets/scanners.py`.
- Extend summary tables and plots in `worldarena/docs/builder.py`.
- Keep dataset roots and exclusions in `config/datasets.yaml`.
- Extend benchmark logic through `worldarena/benchmark/` and `worldarena/generation/`.
- Keep suite weights, metric lists, and backend selection in `config/benchmark.yaml`.
- Keep model runtime contracts in `config/models/*.yaml`.
