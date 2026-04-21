# Benchmark

WorldArena keeps a narrow runnable benchmark core on top of the current curated data roots. The goal is to make the benchmark continuously refinable without splitting the repo into explicit versioned tracks.

## Scope

Official suites:

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

Non-formal assets do not enter official headline reporting. They are reported separately as:

- `auxiliary`
- `experimental`

## Commands

Build the raw inventory:

```bash
uv run worldarena build-inventory --config config/benchmark.yaml
```

Build the manifest:

```bash
uv run worldarena build-manifest --config config/benchmark.yaml
```

Generate outputs with a configured model adapter:

```bash
uv run worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Evaluate predictions:

```bash
uv run worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root /path/to/predictions
```

## Output Contract

Inventory artifacts:

- `artifacts/benchmark/inventory/worldarena_inventory.json`
- `artifacts/benchmark/inventory/worldarena_inventory.csv`
- `artifacts/benchmark/inventory/missing_modalities_report.json`
- `artifacts/benchmark/inventory/count_mismatch_report.csv`

Manifest artifacts:

- `artifacts/benchmark/manifest/worldarena_manifest.jsonl`
- `artifacts/benchmark/manifest/worldarena_manifest_summary.json`
- `artifacts/benchmark/manifest/worldarena_suite_summary.csv`

Run artifacts:

- `artifacts/benchmark/reports/<model_name>/run_manifest.json`
- `artifacts/benchmark/reports/<model_name>/per_sample_scores.jsonl`
- `artifacts/benchmark/reports/<model_name>/per_metric_summary.json`
- `artifacts/benchmark/reports/<model_name>/domain_breakdown.json`
- `artifacts/benchmark/reports/<model_name>/leaderboard.csv`
- `artifacts/predictions/<model_name>/generation_manifest.json`
- `artifacts/predictions/<model_name>/generation_records.jsonl`

## Metric Surface

Default suite metric groups:

- `image_static`: `scene_alignment`, `style_consistency`, `subjective_quality`
- `image_dynamic`: `dynamic_alignment`, `style_consistency`, `subjective_quality`
- `video_static`: `scene_alignment`, `style_consistency`, `subjective_quality`, `rollout_quality`
- `video_dynamic`: `dynamic_alignment`, `style_consistency`, `subjective_quality`, `motion_magnitude`, `motion_smoothness`, `rollout_quality`

The built-in evaluator is backend-aware:

- semantic alignment: `CLIP` when available, otherwise reference-proxy fallback
- style consistency: `VGG19 Gram` when available, otherwise histogram fallback
- subjective quality: `pyiqa` when available, otherwise heuristic quality proxy
- motion metrics: default `Farneback` optical flow backend

To enable the heavier model-backed path:

```bash
uv sync --extra eval
```

To run the first two integrated world models in the same workspace:

```bash
uv sync --python 3.10 --group world_models --extra eval
```

## Prediction Layout

The runner accepts a flat root and tries several common layouts. The simplest is:

```text
predictions/
  image_static/<sample_id>.png
  image_dynamic/<sample_id>.png
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

It also accepts `predictions/<suite>/<split>/<sample_id>.<ext>` and a flat `<sample_id>.<ext>` layout.

## Aggregation

Official benchmark headline scores:

- `S_image_static`
- `S_image_dynamic`
- `S_video_static`
- `S_video_dynamic`

These four suite scores are the official benchmark output. WorldArena does not require a scalar headline total.

Separate non-headline reporting:

- `S_auxiliary`
- `S_experimental`

Optional derived scalar:

- `S_worldarena_derived`

```text
S_worldarena_derived =
  aggregation_profile(
    S_image_static,
    S_image_dynamic,
    S_video_static,
    S_video_dynamic
  )
```

`S_worldarena_derived` is only produced when an explicit `aggregation_profile` is provided. The profile controls how the four official suite scores are combined. This derived scalar is disabled by default and is not part of official headline reporting.

## Adapter Map

The public benchmark surface lives under `worldarena/benchmark/` and `worldarena/generation/`. Current model integration lives under `worldarena/models/`.

- `worldarena/benchmark/`: public benchmark API
- `worldarena/benchmark_core/`: benchmark execution core
- `worldarena/generation/runner.py`: adapter-driven prediction generation
- `worldarena/models/`: model configs and subprocess adapters
- `config/models/lingbot_world.yaml`: LingBot-World runtime contract
- `config/models/matrix_game.yaml`: Matrix-Game runtime contract

The intent is to keep the benchmark contract stable while continuously refining metrics, adapters, and reporting.
