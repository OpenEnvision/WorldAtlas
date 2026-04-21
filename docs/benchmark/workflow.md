# Workflow & Outputs

## Minimal Workflow

`build-manifest` already refreshes benchmark inventory artifacts, so the shortest standard path is:

```bash
uv run worldarena build-manifest --config config/benchmark.yaml
uv run worldarena generate --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
uv run worldarena evaluate --config config/benchmark.yaml --predictions-root artifacts/predictions/lingbot_world
```

One-command path:

```bash
uv run worldarena run-model --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
```

Useful filters:

- `--suites image_static,video_dynamic`
- `--splits dev_public`
- `--limit 32`
- `--overwrite` for regeneration

## Prediction Layout

Accepted layouts include:

```text
predictions/
  image_static/<sample_id>.png
  image_dynamic/<sample_id>.png
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

The evaluator also resolves:

- `predictions/<suite>/<split>/<sample_id>.<ext>`
- `predictions/<suite>/<sample_id>.<ext>`
- `predictions/<split>/<sample_id>.<ext>`
- `predictions/<sample_id>.<ext>`

Supported file types come from the sample modality:

- image: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- video: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

## Output Files

| File | Purpose | Read it first when |
| --- | --- | --- |
| `run_manifest.json` | run metadata, protocol, `missing_predictions`, `reported_metrics`, `reported_diagnostics` | you want to validate the run contract |
| `per_metric_summary.json` | official suite-by-metric averages | you want the headline result |
| `diagnostic_metric_summary.json` | averages for non-official metrics | you want extra but non-headline signals |
| `per_sample_scores.jsonl` | per-sample `raw`, `normalized`, `backend`, `eligibility_status`, `details`, `error` | you need debugging or ablation detail |
| `domain_breakdown.json` | official metrics grouped by benchmark domains | you need structured failure slicing |
| `leaderboard.csv` | flat export of official suite metrics | you want spreadsheet-friendly output |

The evaluation output is only directly comparable when `missing_predictions = 0`.

## Runtime Requirements

Light benchmark metrics run directly from the repo environment. Heavy official video metrics require the extra dependency profile:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
./.venv/bin/python -m spacy download en_core_web_sm
uv run worldarena check-env
```

If the metric source tree is not available under `third_party/`, set:

```bash
export WORLDARENA_METRIC_SOURCE_ROOT=/path/to/compatible-checkout
```

Runtime layout and dependency separation are documented on [Environment](../environment.md) and [Runtime Setup](../third_party.md).
