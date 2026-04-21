# Benchmark

WorldArena keeps a narrow runnable benchmark core on top of the current curated data roots. The goal is to make the benchmark continuously refinable without forcing a single scalar score or a rigid versioned track split.

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
uv run worldarena generate           --config config/benchmark.yaml           --model-config config/models/lingbot_world.yaml
```

Evaluate predictions:

```bash
uv run worldarena evaluate           --config config/benchmark.yaml           --predictions-root /path/to/predictions
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
- `artifacts/benchmark/reports/<model_name>/diagnostic_metric_summary.json` when applicable
- `artifacts/benchmark/reports/<model_name>/domain_breakdown.json`
- `artifacts/benchmark/reports/<model_name>/leaderboard.csv`
- `artifacts/predictions/<model_name>/generation_manifest.json`
- `artifacts/predictions/<model_name>/generation_records.jsonl`

`run_manifest.json` records the official metric columns under `reported_metrics`, any non-official but still computed metrics under `reported_diagnostics`, and the sampling protocol used for the run.

## Metric Surface

Default suite metric groups:

- `image_static`: `prompt_alignment`, `style_consistency`, `perceptual_quality`
- `image_dynamic`: `prompt_alignment`, `style_consistency`, `perceptual_quality`
- `video_static`: `prompt_alignment`, `style_consistency`, `perceptual_quality`
- `video_dynamic`: `prompt_alignment`, `style_consistency`, `perceptual_quality`, `motion_magnitude`, `motion_smoothness`
- `auxiliary`: `reference_appearance_alignment`, `anchor_consistency`
- `experimental`: `reference_appearance_alignment`, `anchor_consistency`, `motion_magnitude`, `motion_smoothness`

Important reporting rule:

- `prompt_alignment` is only official when the sample has explicit instruction supervision.
- For current self-conditioned samples that only carry template-derived prompts, `prompt_alignment` is still computed but marked `eligibility_status = weak_supervision` and excluded from the official aggregate.
- `reference_appearance_alignment` and `anchor_consistency` are diagnostic by design.

## Backend Policy

The evaluator is backend-aware but only official metric paths are allowed into the benchmark aggregate:

- `prompt_alignment`: built-in `CLIPScore` stack by default, with local `CLIP` kept as an explicit diagnostic backend.
- `style_consistency`: built-in `VGG19` gram-matrix stack, reported as backend `vgg19_gram`.
- `perceptual_quality`: built-in `pyiqa` stack combining `clipiqa+` and `laion_aes`, reported as backend `pyiqa_stack`.
- `motion_magnitude` and `motion_smoothness`: official implementation uses calibrated `Farneback` flow statistics.
- Proxy or heuristic fallbacks are not part of the official benchmark path.

The repository can reuse a shared checkpoint cache through `HF_HOME` or a local cache directory such as `cache/hfd`.

To enable the heavier model-backed path inside the project environment:

```bash
uv sync --extra eval --extra quality
```

To run the first two integrated world models in the same workspace:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

## Sampling Protocol

Official evaluation uses `segment_random` sampling. The protocol is deterministic per sample and shared between reference and prediction so the same temporal positions are compared.

`anchor_compat` remains a compatibility protocol for older experiments, but it is not the default official path.

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

## Reporting And Aggregation

WorldArena v1 does not define an official suite-level scalar score and does not define an official benchmark-wide total score.

Official headline output is the per-suite metric vector written to `per_metric_summary.json` and flattened into metric-level columns in `leaderboard.csv`.

Examples:

- `video_static__style_consistency`
- `video_static__perceptual_quality`
- `video_dynamic__motion_magnitude`
- `video_dynamic__motion_smoothness`

Diagnostic metrics stay visible in two places:

- `per_sample_scores.jsonl`, via `eligibility_status`
- `diagnostic_metric_summary.json`, via suite-level diagnostic averages when those metrics produced numeric values

This separation is intentional: metrics that are runnable but not sufficiently supervised for a formal claim should not be silently averaged into official benchmark output.

## Adapter Map

The public benchmark surface lives under `worldarena/benchmark/` and `worldarena/generation/`. Current model integration lives under `worldarena/models/`.

- `worldarena/benchmark/`: public benchmark API
- `worldarena/benchmark_core/`: benchmark execution core
- `worldarena/generation/runner.py`: adapter-driven prediction generation
- `worldarena/models/`: model configs and subprocess adapters
- `config/models/lingbot_world.yaml`: LingBot-World runtime contract
- `config/models/matrix_game.yaml`: Matrix-Game runtime contract

The intent is to keep the benchmark contract stable while continuously refining metrics, adapters, and reporting.
