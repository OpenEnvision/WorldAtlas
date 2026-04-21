# Motion Magnitude

Measures how much motion the predicted video contains under the official motion backend.

## Used By

- `video_dynamic`
- `experimental`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |

The official benchmark path does not require reference media or annotations for this metric.

## Raw Output

- Official backend: `SEA-RAFT`.
- Raw output is one scalar motion-magnitude statistic from the official flow pipeline.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | native motion-magnitude score |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_count` | decoded frame count |
| `details.native_score` | same scalar recorded in details |

## Normalization

Current z-score config:

| Parameter | Value |
| --- | ---: |
| `avg` | `3.2425` |
| `std` | `3.4505` |
| `z_min` | `-0.8638` |
| `z_max` | `2.7498` |
| output range | `[0.25, 0.75]` |
| direction | higher is better |

Higher raw motion magnitude maps to a higher normalized score until it reaches the configured upper clip.

## Official Rule

Official whenever it returns a numeric value in official suites. It also appears diagnostically in `experimental`.

## How To Read It

- High: the video contains stronger motion under the benchmark calibration.
- Low: the rollout is closer to static or under-moves relative to the benchmark reference range.

## Common Misreads

- This metric does not judge whether the motion is correct.
- It does not measure smoothness.
- A sample can have high motion magnitude and still fail `motion_accuracy`.

## Implementation

- `worldarena/benchmark/metrics/motion.py`
- `worldarena/benchmark/official_runtime.py`
