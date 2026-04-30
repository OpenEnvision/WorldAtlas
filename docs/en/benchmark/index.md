# Benchmark Overview

WorldArena reports a suite-specific metric vector for controllable world generation. The contract stays intentionally narrow: fixed manifest shape, fixed evaluation protocol, explicit sample-level eligibility rules, and no benchmark-wide total score.

The repository architecture is broader than the current public benchmark surface. Model configs and manifests already carry task metadata for future 3D, 4D, reconstruction, and simulator-backed tracks, but the current headline benchmark still centers on the four formal image/video suites below. See [Architecture](../architecture.md).

## Formal Suites

| Suite | Modality | Conditioning contract | Official metrics when eligible | Diagnostic metrics and notes |
| --- | --- | --- | --- | --- |
| `image_static` | image -> video | `reference_image`, 1 frame; prediction is an I2V rollout | video-output metric superset. Common official metrics include `prompt_alignment`*, `object_detection`*, `style_consistency`, `perceptual_quality`, `reprojection_error`, `optical_flow_aepe`, `motion_magnitude`, `motion_smoothness`, and `physical_fps_*`; pose/mask/reference-video metrics score only when compatible sidecars exist | `semantic_alignment`, `brightness_consistency`, `color_temperature_consistency`, `sharpness_retention`, `depth_accuracy`, `depth_collision`, `psnr`, `ssim`, `met3r_consistency`, `reconstruction_consistency`, `long_sequence_*`; reference-video metrics are usually unavailable for pure image references |
| `image_dynamic` | image -> video | `reference_image`, 1 frame; prediction is an I2V rollout | same as `image_static`, with more dynamic/object supervision when available | same as `image_static` |
| `video_static` | video | `first_frame`, 1 frame | `prompt_alignment`*, `style_consistency`, `perceptual_quality`, `camera_error`, `seen_region_preservation`, `novel_region_fidelity`, `object_detection`*, `reprojection_error`, `optical_flow_aepe` | `semantic_alignment`, `brightness_consistency`, `color_temperature_consistency`, `sharpness_retention`, `depth_accuracy`, `depth_collision`, `psnr`, `ssim`, `met3r_consistency`, `reconstruction_consistency` |
| `video_dynamic` | video | `reference_video`, prefix rollout continuation | `prompt_alignment`*, `style_consistency`, `perceptual_quality`, `object_permanence`, `pmf`, `motion_accuracy`, `motion_magnitude`, `motion_smoothness`, `physical_fps_error`, `physical_fps_pct_error`, `physical_fps_intra_video_cv` | `semantic_alignment`, `brightness_consistency`, `color_temperature_consistency`, `sharpness_retention`, `depth_accuracy`, `depth_collision`, `psnr`, `ssim`, `met3r_consistency`, `reconstruction_consistency`, `long_sequence_*` |
| `experimental` | mixed | mixed | `motion_magnitude`, `motion_smoothness` when this supplemental suite is evaluated explicitly; not a formal headline suite | `memory_revisit_consistency` |

\* Sample-level eligibility still applies. `prompt_alignment` is computed only with explicit instruction supervision and non-self conditioning; `object_detection` is computed only from explicit structured object annotations. Other cases are `not_applicable` and produce no score. See [Protocol & Eligibility](protocol.md).

`video_dynamic` uses the benchmark's prefix-continuation contract: by default the first `25%` of the reference rollout, with a floor of `4` frames, is treated as conditioning input and the continuation region becomes the evaluation target.

## Six Evaluation Pillars

WorldArena now groups benchmark metrics into six intentionally broad evaluation pillars. They should be read as capability-level buckets for benchmark coverage, not as narrow backend-specific bins:

| Dimension key | User-facing label | Scope |
| --- | --- | --- |
| `long_sequence` | Long sequence (memory) | state retention, appearance persistence, and long-horizon rollout stability |
| `action_control` | Action control | scene control, camera control, and object/action control |
| `consistency_3d_4d` | 3D/4D consistency | geometry, multiview, reprojection, and motion-field consistency |
| `physics` | Physics | motion realism plus simulator-backed physics metrics |
| `quality` | Quality | style, perceptual quality, low-level appearance stability, and reconstruction-style fidelity |
| `real_time` | Real time | tempo and FPS calibration |

`per_dimension_summary.json` is the official suite-level roll-up over those six pillars. See [Metrics Overview](metrics/index.md).

## Headline Reporting Contract

- The primary result is `per_metric_summary.json`: per-suite averages over numeric outputs with `eligibility_status = official`.
- `per_dimension_summary.json` is the official six-dimension roll-up over the same eligible metric outputs.
- `leaderboard.csv` is only a flattened export of that same official suite-by-metric vector.
- `diagnostic_metric_summary.json` is intentionally separate. It collects only metrics that are explicitly diagnostic.
- `not_applicable_summary.json` records metrics skipped because strong supervision is absent; these entries are neither failures nor interpretable scores.
- A configured metric is not automatically a headline metric. It must both run successfully and satisfy the relevant eligibility gate for that sample.
- WorldArena does not define a benchmark-wide scalar and does not define an official single-number suite score.

## Separate Physics Configuration

`config/benchmark_physics.yaml` defines a simulator-backed physics config under track `physics_i2v`. It reuses the suite label `experimental`, but it is not the same thing as the diagnostic-only `experimental` slice in the default `config/benchmark.yaml`.

- physics runs still write their own `per_metric_summary.json`, `leaderboard.csv`, `per_sample_scores.jsonl`, and failure reports
- the reported metrics are simulator-backed quantities such as `mse`, `st_iou`, `depth_warp_charb`, `chamfer_4d`, and `worldline_*`
- those results stay outside the four-suite public vectors above and should not be mixed with them

See [Metrics / Physics](metrics/dimensions/physics.md) and [Simulator-Backed Physics](metrics/physics/index.md).

## Documentation Map

- [Protocol & Eligibility](protocol.md): sampling, conditioning, normalization, and official vs diagnostic rules.
- [Workflow & Outputs](workflow.md): commands, prediction layout, runtime resolution, report files, and physics-manifest workflows.
- [Metrics / Physics](metrics/dimensions/physics.md): the unified entry point for physical motion realism and simulator-backed physics metrics.
- [Metrics Overview](metrics/index.md): the full metric surface from `config/benchmark.yaml`, grouped by dimension and suite.
- [Runtime Setup](../thirdparty.md): cache layout, third-party checkouts, and checkpoint integration.
