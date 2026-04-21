# Optical Flow AEPE

Measures photometric motion consistency through average endpoint error under the official optical-flow backend.

## Used By

- `video_static`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |

Reference media is not required by the official path.

## Raw Output

- Official backend: `SEA-RAFT`.
- Raw output is one scalar `AEPE` value.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | average endpoint error |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_count` | decoded frame count |
| `details.native_score` | same scalar recorded in details |

## Normalization

Current config:

- `empirical_min = 0.0`
- `empirical_max = 1.192`
- `higher_is_better = false`

Lower raw error becomes higher normalized score.

## Official Rule

Official whenever it returns a numeric value.

## How To Read It

- High normalized score: adjacent-frame motion is photometrically consistent under the official flow estimator.
- Low normalized score: temporal transitions are noisy, discontinuous, or artifact-heavy.

## Common Misreads

- This is not target motion correctness.
- It says nothing about whether the intended object moved in the intended way.
- It is a consistency metric, not a semantic control metric.

## Implementation

- `worldarena/benchmark/metrics/consistency.py`
- `worldarena/benchmark/official_runtime.py`
