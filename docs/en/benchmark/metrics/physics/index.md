# Simulator-Backed Physics

WorldArena keeps simulator-backed physics metrics under the `physics` dimension because they compare generated videos against exported simulator state, not only against rendered reference media. The physics track reuses the same manifest, report, and CLI infrastructure as the main benchmark, but it uses its own config, manifests, reports, and metric contract.

Within the broader six-pillar benchmark framing, this entire track should be read as the high-fidelity end of the `physics` pillar rather than as a separate conceptual taxonomy.

!!! note "Do not conflate the two experimental suites"
    `config/benchmark.yaml` contains a diagnostic-only `experimental` suite. `config/benchmark_physics.yaml` also uses the suite label `experimental`, but there it names the simulator-backed physics track and its metrics are aggregated into that run's `per_metric_summary.json`.

## Config Identity

| Item | Current value |
| --- | --- |
| Config | `config/benchmark_physics.yaml` |
| Suite | `experimental` |
| Track | `physics_i2v` |
| Modality | video |
| Conditioning strategy | `external_image` |
| Metric backend | `simulator_reference` for every configured metric |
| Official protocol | `segment_random` |
| Compatibility protocol | `anchor_compat` |
| Target FPS | `24` |
| Configured artifact roots | `artifacts/benchmark/physics/inventory`, `artifacts/benchmark/physics/manifest`, `artifacts/benchmark/physics/reports`, `.cache/worldarena/benchmark_physics` |

This track is still evaluated with the standard `worldarena evaluate` command. What changes is the manifest content: each sample carries a simulator-specific `physics_spec` and `physics_case` payload that point the evaluator at exported `world_gt` artifacts and the external physics pipeline checkout.

## Sample Sources

WorldArena currently supports three ways to materialize physics samples:

1. `build-physics-manifest` from an exported simulator pipeline root.
   It expects `assets/<sample_id>.mp4` and `assets/<sample_id>.json`, exports the first frame as the conditioning image when needed, and writes sample ids like `physics__<source_sample_id>__i2v`.
2. `build-physics-seed-manifest` from a seed-image manifest plus an optional case catalog.
   It creates image-conditioned continuation cases with ids like `physics_seed__<case_id>__i2v`.
3. Python-only online helpers such as `build_online_physics_sample(...)` and `evaluate_online_physics_i2v(...)`.
   These are useful for ad hoc i2v debugging outside the batch CLI path.

## Manifest Payload

Physics samples add two important metadata blocks on top of the standard benchmark schema.

`physics_spec` describes the contract that should hold for a generated video:

- benchmark family and backend name
- source sample id and prompt metadata
- `metrics_design` grouped into `video_physics`, `geometry_3d`, `temporal_consistency`, and `world_4d`
- `optional_metrics_design` for backend outputs that are recognized but not enabled by default
- `metric_contract` such as `required_core_metrics`, `require_full_frame_coverage`, and `require_pred_world_artifacts`
- case taxonomy and simulator hints copied from the source metadata

`physics_case` describes where runtime artifacts live:

- `physics_pipeline_root` and resolved `physics_data_root`
- local case root, simulator root, and `submitted_video_path`
- `metadata_template_path` and inlined metadata template payload
- local `world_gt_root`
- export benchmark name and render split used by the external simulator scripts

## Reporting Behavior

Physics runs reuse the same output file contract as the main benchmark:

- `run_manifest.json`
- `per_metric_summary.json`
- `per_dimension_summary.json`
- `per_sample_scores.jsonl`
- `metric_failures.json`
- `leaderboard.csv`

Important differences:

- the reported suite is `experimental`, not one of the four public formal suites
- physics metrics default to `eligibility_status = official` when a numeric value is returned, so `per_metric_summary.json` becomes the main physics summary
- `per_dimension_summary.json` now reports a single top-level `physics` roll-up over the enabled simulator-backed metrics
- those numbers should not be mixed with or compared directly against the public four-suite vectors from `config/benchmark.yaml`

## Read Next

- [Physics Workflow](workflow.md): required external checkout, manifest build commands, evaluation lifecycle, and readiness states
- [Physics Metrics & Contracts](metrics.md): metric groups, normalization ranges, required artifacts, and success criteria
- [Workflow & Outputs](../../workflow.md): the shared benchmark report layout used by both the public benchmark and the physics track
