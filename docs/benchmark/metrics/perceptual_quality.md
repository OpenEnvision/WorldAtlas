# Perceptual Quality

Measures perceived visual quality of the prediction itself, without comparing it to the reference rollout.

## Used By

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

## Inputs

| Input | Source |
| --- | --- |
| prediction anchor frames | sampled under the official protocol |

Reference media is not used.

## Raw Output

- Official backend: `pyiqa_stack`.
- The reference runtime combines two quality components:
  - `clipiqa+`
  - `laion_aes`
- Each anchor frame gets a normalized quality score per component.
- The frame score is the mean of required components.
- The final metric is the mean across anchor frames.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | mean frame quality score |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_scores` | per-anchor mean quality scores |
| `details.components` | per-component frame scores and native ranges |

## Normalization

Current config:

- `lower = 0.0`
- `upper = 1.0`
- `higher_is_better = true`

As configured now, benchmark normalization is identity on the reference output.

## Official Rule

`perceptual_quality` is official whenever it returns a numeric value.

## How To Read It

- High: sharper, cleaner, more aesthetically plausible frames.
- Low: blur, noise, artifacts, compression damage, or unstable local structure.

## Common Misreads

- This is not a semantic alignment metric.
- It can score a beautiful but incorrect sample highly.
- It is anchor-based, so it is not a full temporal smoothness metric.

## Implementation

- `worldarena/benchmark/metrics/quality.py`
- `worldarena/benchmark/reference_runtime.py`
