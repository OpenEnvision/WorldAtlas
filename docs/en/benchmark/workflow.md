# Workflow & Outputs

All commands below assume the project's conda environment is already activated.

## Main Benchmark Workflow

`build-manifest` already refreshes benchmark inventory artifacts, so the shortest standard path is:

```bash
worldarena build-manifest --config config/benchmark.yaml
worldarena generate --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
worldarena evaluate --config config/benchmark.yaml --predictions-root artifacts/predictions/lingbot_world
```

One-command path:

```bash
worldarena run-model --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
```

Use `build-inventory` only when you explicitly want the raw asset audit without immediately writing a manifest:

```bash
worldarena build-inventory --config config/benchmark.yaml
```

## Useful Filters And Overrides

- `--suites image_static,video_dynamic`
- `--limit 32`
- `--overwrite` for regeneration
- `--manifest /path/to/worldarena_manifest.jsonl` when you want to pin a specific manifest snapshot
- `--model-config ...` on `evaluate` when model-specific metric overrides should be applied before scoring

## Prediction Layout

Accepted layouts include:

```text
predictions/
  image_static/<sample_id>.mp4
  image_dynamic/<sample_id>.mp4
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

The evaluator also resolves:

- `predictions/<suite>/<sample_id>.<ext>`
- `predictions/<sample_id>.<ext>`

Supported file types follow the benchmark prediction contract. All formal suites currently evaluate video predictions; `image_static` and `image_dynamic` are image-conditioned video rollouts.

- video: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- non-formal or future image artifacts: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

## Output Files

| File | Purpose | Read it first when |
| --- | --- | --- |
| `run_manifest.json` | run metadata, protocol, `missing_predictions`, `reported_metrics`, `reported_diagnostics` | you want to validate the run contract |
| `per_metric_summary.json` | official suite-by-metric averages | you want the headline result |
| `per_dimension_summary.json` | official suite-by-dimension averages over the six top-level benchmark dimensions | you want the compact dimension roll-up |
| `diagnostic_metric_summary.json` | averages for diagnostic metrics | you want extra but non-headline analysis signals |
| `not_applicable_summary.json` | counts and reasons for metrics skipped because strong supervision is absent | you want to verify which metrics were ineligible |
| `per_sample_scores.jsonl` | per-sample `raw`, `normalized`, `backend`, `eligibility_status`, `details`, `error` | you need debugging or ablation detail |
| `domain_breakdown.json` | official metrics grouped by benchmark domains | you need structured failure slicing |
| `leaderboard.csv` | flat export of official suite metrics | you want spreadsheet-friendly output |

Runs are only directly comparable when `missing_predictions = 0` and the same suites, manifest snapshot, and metric config were used.

## Official Comparison Rule

For a leaderboard-style comparison across several models:

- build one manifest snapshot and reuse it for all models
- keep the same suites, manifest snapshot, and metric configuration across all compared runs

## Simulator-Backed Physics Workflows

The simulator-backed physics track uses a separate config and manifest root. The full contract now lives under Metrics / Physics: [Physics](metrics/dimensions/physics.md), [Simulator-Backed Physics](metrics/physics/index.md), [Physics Workflow](metrics/physics/workflow.md), and [Physics Metrics & Contracts](metrics/physics/metrics.md). The short operational path is:

Build a manifest from an exported physics pipeline:

```bash
worldarena build-physics-manifest \
  --physics-pipeline-root /path/to/physics-pipeline
```

Evaluate predictions against that manifest:

```bash
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

Build an image-seed physics manifest for i2v continuation cases:

```bash
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --physics-pipeline-root /path/to/physics-pipeline
```

Those commands keep the same report contract, but they target the simulator-backed `experimental` suite instead of the four formal headline suites.

Practical notes:

- the default report root is `artifacts/benchmark/physics/reports/<model_name>` when `--output-dir` is omitted
- prediction discovery uses the same resolver as the main benchmark, so files can live under `experimental/` or the prediction root itself
- per-sample details include `physics_status`, `world_gt_status`, `pred_world_status`, `metric_contract`, and `missing_world_artifacts`
