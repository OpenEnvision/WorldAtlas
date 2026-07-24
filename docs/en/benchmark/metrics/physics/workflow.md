# Physics Workflow

All commands below assume the project's conda environment is already activated.

## Required External Checkout

The simulator-backed path depends on an external physics-pipeline checkout passed explicitly as `--physics-pipeline-root`.

| Path or file | Why it is needed |
| --- | --- |
| `assets/<sample_id>.mp4` | reference rollout used to define the case and export the default conditioning frame |
| `assets/<sample_id>.json` | prompt metadata, object taxonomy, scene tags, and simulator hints |
| `benchmark_export.py` | exports `world_gt` artifacts into the pipeline data root |
| `benchmark_eval.py` | computes the simulator-backed metrics from `world_gt`, the generated video, and optional `pred_world` artifacts |
| `main_part2.py` | reconstructs `pred_world` artifacts for generated videos when the current metric contract requires them |

The current core contract assumes all three scripts exist. Missing `benchmark_export.py` or `benchmark_eval.py` marks a case as `missing_backend`; missing `main_part2.py` usually surfaces later as a `pred_world` contract failure.

## Path Resolution

- `build-physics-manifest` and `build-physics-seed-manifest` record the explicit `physics_pipeline_root` into each sample's `physics_case`.
- The physics data root defaults to `<physics_pipeline_root.parent.parent>/.runtime/physics_pipeline/data_root`.
- The benchmark runner then stages reports under `artifacts/benchmark/physics/reports/<model_name>` unless `--output-dir` overrides it.

Prediction discovery uses the normal benchmark resolver, so any of these layouts are accepted for physics runs:

- `predictions/experimental/<sample_id>.mp4`
- `predictions/<sample_id>.mp4`

## Build A Manifest From Simulator Assets

Standard batch path:

```bash
worldarena build-physics-manifest \
  --physics-pipeline-root /path/to/physics-pipeline
```

The default output root is `artifacts/benchmark/physics/manifest` and contains:

```text
artifacts/benchmark/physics/manifest/
  conditioning/<source_sample_id>.png
  cases/physics__<source_sample_id>__i2v/
    simulator/
      metadata_template.json
      submitted_video.mp4
    world_gt/
  worldarena_physics_manifest.jsonl
  worldarena_physics_manifest_summary.json
```

Notes:

- `conditioning/<source_sample_id>.png` is exported from the first frame of the reference rollout when that PNG is not already present
- the summary JSON records item counts and the current `required_core_metrics`
- sample ids are deterministic: `physics__<source_sample_id>__i2v`

## Evaluate Predictions

```bash
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

Per-sample evaluation follows this lifecycle:

1. Stage the submitted prediction into the case-local `simulator/submitted_video.mp4` path.
2. Inspect the case with `inspect_physics_case(...)`.
3. If local `world_gt` is incomplete, try to export it from the physics pipeline or sync an already exported copy.
4. If the local reference video is still missing, fail early with `physics_status`, remediation hints, and missing artifact names.
5. If automatic `pred_world` generation is enabled, run `main_part2.py` and export the predicted world artifacts.
6. Run `benchmark_eval.py` and write `<prediction_stem>_physics_report.json`.
7. Attach fields such as `physics_status`, `world_gt_status`, `pred_world_status`, `metric_contract`, and `missing_world_artifacts` to the benchmark output.

## Readiness States

`inspect_physics_case(...)` reports one of the following statuses before or after auto-remediation:

| Status | Meaning |
| --- | --- |
| `ready` | local `world_gt` already contains the full required metric artifact set |
| `needs_sync` | a complete exported `world_gt` exists under the physics data root and can be synced locally |
| `missing_backend` | required pipeline scripts such as `benchmark_export.py` or `benchmark_eval.py` are absent |
| `partial_world_gt` | local `world_gt` contains at least `rgb.mp4`, but the full metric artifact set is incomplete |
| `needs_export` | the evaluator must export `world_gt` before the case can satisfy the metric contract |

`preflight_physics_benchmark(...)` wraps the same checks for a batch of samples and records missing metric artifacts plus remediation commands.

## Seed-Image Manifest Flow

The seed-image path turns generated seed images into simulator-backed continuation cases:

```bash
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --catalog /path/to/physics-seed-catalog.json \
  --physics-pipeline-root /path/to/physics-pipeline
```

Selection rules:

- each input record must expose a non-empty `case_id`
- only records with `status = ok` are kept
- `--case-ids` can further restrict the selected cases
- if `--catalog` is provided, the catalog is merged by `case_id` first and the image-manifest record wins on key conflicts

The generated sample ids are `physics_seed__<case_id>__i2v`. The builder also carries forward structured fields such as motion family, interaction family, difficulty tier, support geometry, object metadata, and simulator overrides into the resulting `physics_spec` / `physics_case`.

## Python-Only Online Path

For ad hoc debugging you can skip the batch manifest CLI and evaluate a single i2v case directly from Python:

```python
from worldarena.benchmark.physics_backend import evaluate_online_physics_i2v

result = evaluate_online_physics_i2v(
    prompt="A toy ball rolls down a wooden incline.",
    image_path="tmp/seed.png",
    generated_video_path="tmp/pred.mp4",
    output_root="tmp/physics_online",
    physics_pipeline_root="/path/to/physics-pipeline",
)
print(result["success"], result["report_path"])
```

This path builds an `online_physics__...` sample, runs the external simulator pipeline, exports `pred_world`, and returns the same style of result payload used by the batch evaluator.
