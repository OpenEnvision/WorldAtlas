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

WorldArena keeps four formal suites as the public core:

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

Official headline reporting is a per-suite metric vector, not a suite-level scalar and not a benchmark-wide total score.

Current official metric families are:

- `image_static`: `style_consistency`, `perceptual_quality`
- `image_dynamic`: `style_consistency`, `perceptual_quality`
- `video_static`: `style_consistency`, `perceptual_quality`
- `video_dynamic`: `style_consistency`, `perceptual_quality`, `motion_magnitude`, `motion_smoothness`

`prompt_alignment` is still computed, but on self-conditioned samples with template-derived prompts it is reported as `weak_supervision` and routed to diagnostics instead of the official aggregate.

Non-formal assets stay outside official headline reporting and are routed into separate `auxiliary` or `experimental` reporting.

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
uv run worldarena generate           --config config/benchmark.yaml           --model-config config/models/lingbot_world.yaml           --limit 8
```

```bash
uv run worldarena generate           --config config/benchmark.yaml           --model-config config/models/matrix_game.yaml           --suites video_static,video_dynamic           --limit 8
```

### Evaluate Predictions

For official metric backends, sync the benchmark extras:

```bash
uv sync --extra eval --extra quality
```

For the integrated world models, install the runtime group as well:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

If the machine already has a CUDA-enabled system Python with `torch`, the model configs can reuse it through a local runtime shim. The configs default to `python_bin: auto`, which prefers `.runtime312/bin/python` when present.

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python           diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg           opencv-python==4.10.0.84
```

Optional model-side environment overrides live under `model.env`. This is the cleanest place to pin `CUDA_VISIBLE_DEVICES`, `HF_HOME`, proxy settings, or other runtime variables without changing the benchmark runner itself.

A shared checkpoint cache can be mounted through `cache/hfd`. In the current remote setup, `cache/hfd` points to the shared WorldEvals cache and is used by reference metric backends such as VFI, optical flow, SAM, and GroundingDINO.

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
uv run worldarena run-model           --config config/benchmark.yaml           --model-config config/models/lingbot_world.yaml           --limit 8
```

The runner writes:

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `diagnostic_metric_summary.json` when any metric is non-official for eligibility reasons
- `domain_breakdown.json`
- `leaderboard.csv`
