# Reprojection Error

Measures geometric self-consistency of the predicted video.

## Used By

- `video_static`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |

Reference media and annotation-side supervision are not required by the official score path.

## Raw Output

- Official backend: `DROID-SLAM`.
- Raw output is a single reprojection error scalar.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | reprojection error |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_count` | decoded frame count |
| `details.native_score` | same scalar recorded in details |

## Normalization

Current config:

- `empirical_min = 0.0`
- `empirical_max = 1.0719`
- `higher_is_better = false`

Lower raw error becomes higher normalized score.

## Official Rule

Official whenever it returns a numeric value.

## How To Read It

- High normalized score: the predicted video is geometrically easier to reconstruct and reproject consistently.
- Low normalized score: geometry is unstable, contradictory, or heavily artifacted.

## Common Misreads

- This is not prompt alignment.
- It is not the same as camera control.
- A video can be geometrically consistent but semantically wrong.

## Implementation

- `worldarena/benchmark/metrics/consistency.py`
- `worldarena/benchmark/official_runtime.py`
