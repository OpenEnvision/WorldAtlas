# Protocol & Eligibility

## Evaluation Protocol

| Item | Value | Why it matters |
| --- | --- | --- |
| Official protocol | `segment_random` | reference and prediction are sampled at the same temporal positions |
| Compatibility protocol | `anchor_compat` | legacy path for older experiments, not the default headline protocol |
| Target FPS | `24` | normalizes evaluation-time temporal assumptions |
| Anchor positions | `0.0`, `0.3333`, `0.6667`, `1.0` | anchor-based metrics read these positions from the rollout |
| Motion frame count | `12` | motion metrics sample 12 frames per evaluated item |
| Segment count | `4` | controls deterministic segment sampling under `segment_random` |
| Seed | `17` | keeps sampling deterministic at the sample level |

Anchor-based metrics and motion metrics share the same sample-local protocol state, so reference and prediction are compared on aligned temporal support.

## Eligibility Status

| Status | Included in official aggregates | Typical reason |
| --- | --- | --- |
| `official` | yes | metric has numeric output and the sample satisfies supervision requirements |
| `weak_supervision` | no | the metric is computed, but the supervision source is too weak for a headline claim |
| `diagnostic` | no | the metric is intentionally analysis-only |
| `normalized = null` | no | the metric could not produce a numeric value for that sample |

## Metric Gates

| Metric | Official gate |
| --- | --- |
| `prompt_alignment` | sample must have explicit instruction supervision and must not be self-conditioned |
| `object_detection` | official only when object prompts come from explicit annotation fields |
| `camera_error` | requires annotation-side camera poses |
| `motion_accuracy` | requires segmentation masks |
| `reference_appearance_alignment` | never official |
| `anchor_consistency` | never official |
| all others | official whenever the metric returns a numeric value |

Two important consequences:

- Self-conditioned image samples can still compute `prompt_alignment`, but only as `weak_supervision`.
- Caption-derived object prompts are useful for debugging, but they do not enter official `object_detection` aggregates.

## Normalization

All benchmark headline values are stored as `normalized` scores in `[0, 1]`.

| Metric family | Raw form | Normalization rule |
| --- | --- | --- |
| bounded scalar metrics | one scalar already bounded or naturally mapped to a bounded interval | min/max scaling from `config/benchmark.yaml` |
| empirical scalar metrics | one scalar where lower is better or higher is better | empirical min/max scaling, optionally inverted |
| z-score metrics | one scalar calibrated against benchmark-wide statistics | z-score mapping into a configured sub-range of `[0, 1]` |
| component metrics | multiple raw components | normalize each component, then reduce |

Current benchmark-specific reducers:

- `camera_error`: normalize rotation and translation separately, then take a geometric mean.
- `motion_smoothness`: normalize `MSE`, `SSIM`, and `LPIPS` separately, then take an arithmetic mean.

## Aggregation Rules

- `per_metric_summary.json` averages only numeric `normalized` values with `eligibility_status = official`.
- `diagnostic_metric_summary.json` averages numeric outputs that were computed but excluded from official reporting.
- `domain_breakdown.json` is also official-only. It slices official scores by `style`, `environment`, `scene`, and `motion_category`.
- Missing predictions are counted in `run_manifest.json`, but they do not become zeros. They remain missing.

Compare runs only when the same suites, splits, protocol, and prediction completeness were used.
