# Style Consistency

Measures how well the prediction preserves the reference rollout's global appearance and style.

## Used By

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

## Inputs

| Input | Source |
| --- | --- |
| prediction anchor frames | sampled under the official protocol |
| reference anchor frames | sampled from the benchmark reference at the same positions |

## Raw Output

- Official backend: `VGG19` Gram-matrix reference path.
- Each prediction anchor frame is paired with the reference anchor at the same temporal position.
- Native style distance is converted to a similarity score with `1 / (1 + distance)`.
- The final score is the mean similarity across anchor pairs.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | mean style similarity across anchor pairs |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_scores` | per-anchor similarity values |
| `details.native_frame_distances` | per-anchor Gram-distance values |
| `details.native_layer_distances` | layer-level style distances used by the reference runtime |

## Normalization

Current config:

- `lower = 0.0`
- `upper = 1.0`
- `higher_is_better = true`

The current normalization is effectively identity because the reference path already emits a bounded similarity score.

## Official Rule

`style_consistency` is official whenever it returns a numeric value.

## How To Read It

- High: color palette, texture statistics, and overall visual style stay close to the reference rollout.
- Low: the result may still be semantically correct, but it drifts in rendering style or appearance structure.

## Common Misreads

- This is not a prompt-following metric.
- It can reward close appearance matching even when motion control is weak.
- High style consistency does not imply high perceptual quality.

## Implementation

- `worldarena/benchmark/metrics/style.py`
- `worldarena/benchmark/reference_runtime.py`
