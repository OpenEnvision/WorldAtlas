# Protocol & Eligibility

## Sampling Contract

| Item | Value | Why it matters |
| --- | --- | --- |
| Official protocol | `segment_random` | headline runs sample reference and prediction on matched temporal support |
| Compatibility protocol | `anchor_compat` | legacy anchor-only path for older experiments |
| Target FPS | `24` | normalizes evaluation-time temporal assumptions |
| Anchor positions | `0.0`, `0.3333`, `0.6667`, `1.0` | anchor-based metrics read these positions from the rollout window |
| Motion frame count | `12` | motion-oriented metrics decode a 12-frame window per sample |
| Segment count | `4` | deterministic segment selection under `segment_random` |
| Seed | `17` | keeps sampling stable at the sample level |

Anchor-based metrics and motion metrics share the same sample-local protocol state, so prediction and reference are compared on aligned temporal support.

## Conditioning Contract

| Asset type | Conditioning strategy | Default parameters | Evaluation implication |
| --- | --- | --- | --- |
| image suites | `reference_image` | `conditioning_frame_count = 1` | a still reference image is the conditioning signal; prediction is parsed as a video rollout, so prediction-side video metrics can run |
| `video_static` | `first_frame` | `conditioning_frame_count = 1` | the first frame is given, and the rollout is judged as a camera-conditioned continuation |
| `video_dynamic` | `reference_video` | `dynamic_prefix_ratio = 0.25`, `dynamic_prefix_min_frames = 4` | the prefix becomes conditioning input and the continuation region becomes the target rollout |

The manifest stores this contract per sample, so downstream generation and evaluation do not need to infer it from the file layout.

## Eligibility Status

| Status | Included in official aggregates | Typical reason |
| --- | --- | --- |
| `official` | yes | the metric produced a numeric value and the sample satisfied the supervision gate |
| `diagnostic` | no | the metric is intentionally analysis-only |
| `not_applicable` | no | the sample lacks strong supervision for that metric, so no score is computed or aggregated |
| `normalized = null` | no | runtime failure, missing input, or an inapplicable metric |

## Metric Gates

| Metric or family | Official gate |
| --- | --- |
| `prompt_alignment` | sample must have explicit instruction supervision and must not be self-conditioned |
| `object_detection` | computed only when object prompts come from explicit structured annotations; caption-derived prompts are not scored |
| `camera_error` | requires pose annotations under `sample.annotation_path` |
| `motion_accuracy` | requires aligned segmentation masks |
| `object_permanence` | requires aligned dynamic masks plus a readable reference video |
| `physical_fps_*` | require a working `Pulse-of-Motion` runtime and usable container FPS or manifest FPS |
| `depth_collision` | never official; diagnostic only. Requires a video prediction and Video-Depth-Anything runtime |
| `reconstruction_consistency` | never official; diagnostic only. Requires a video prediction, Depth-Anything-3, and LPIPS |
| `memory_revisit_consistency` | never official; diagnostic only. Intended for revisit / loop-closing experimental samples |
| `semantic_alignment`, `subject_consistency`, `background_consistency`, `depth_accuracy`, `psnr`, `ssim`, `met3r_consistency`, `long_sequence_*` | never official; diagnostic only |
| `anchor_consistency` | never official; diagnostic only |
| all remaining metrics in the formal suites | official whenever the metric returns a numeric value |

Two common consequences:

- Self-conditioned samples, or samples without explicit instruction labels, do not compute `prompt_alignment`; they are listed in `not_applicable_summary.json`.
- Caption-derived object prompts are not used for `object_detection`; samples without explicit object annotations are `not_applicable`.
- `image_static` and `image_dynamic` predictions are I2V videos. Prediction-only video metrics run normally; metrics requiring reference video, pose, or masks return unavailable when those sidecars are missing.

## Normalization

All headline values are stored as `normalized` scores in `[0, 1]`.

| Metric family | Raw form | Normalization rule |
| --- | --- | --- |
| bounded scalar metrics | one scalar already bounded or naturally mapped to a bounded interval | direct min/max scaling from `config/benchmark.yaml` |
| empirical scalar metrics | one scalar where lower is better or higher is better | empirical min/max scaling, optionally inverted |
| z-score metrics | one scalar calibrated against benchmark-wide statistics | z-score mapping into a configured sub-range of `[0, 1]` |
| component metrics | multiple raw components | normalize each component, then reduce |

Current benchmark-specific reducers:

- `camera_error`: normalize rotation and translation separately, then reduce with a geometric mean.
- `motion_smoothness`: normalize `MSE`, `SSIM`, and `LPIPS` separately, then reduce with an arithmetic mean.

## Aggregation Rules

- `per_metric_summary.json` averages only numeric `normalized` values with `eligibility_status = official`.
- `per_dimension_summary.json` regroups those same official outputs into six top-level dimensions: `long_sequence`, `action_control`, `consistency_3d_4d`, `physics`, `quality`, and `real_time`.
- `diagnostic_metric_summary.json` averages only metrics that are explicitly designed as diagnostics.
- `not_applicable_summary.json` records metrics skipped because the sample lacks strong supervision. These entries are not failures and not diagnostic scores.
- `domain_breakdown.json` is also official-only. It slices official scores by `style`, `environment`, `scene`, and `motion_category`.
- Missing predictions are counted in `run_manifest.json`, but they do not turn into zeros. They remain missing.

The same report contract is reused by the simulator-backed physics configuration, but those runs are kept on a separate manifest and a separate config namespace. That physics config also uses the suite label `experimental`, yet there it names the simulator-backed track rather than the diagnostic-only `experimental` slice from `config/benchmark.yaml`. See [Metrics / Physics](metrics/dimensions/physics.md).

Compare runs only when the same suites, protocol, metric config, and prediction completeness were used. WorldArena currently has a single test set; the manifest `split` field is fixed to `test` and kept only as a compatibility field.
