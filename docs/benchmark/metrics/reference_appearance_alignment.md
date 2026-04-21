# Reference Appearance Alignment

Measures appearance similarity between prediction and reference at the anchor frames. It is a diagnostic metric, not an official benchmark headline metric.

## Used By

- `auxiliary`
- `experimental`

## Inputs

| Input | Source |
| --- | --- |
| prediction anchor frames | sampled under the official protocol |
| reference anchor frames | sampled at the same positions |

## Raw Output

- Backend: `CLIP` image-image similarity.
- Each prediction anchor is paired with the reference anchor at the same temporal position.
- The metric averages the per-anchor similarities.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | mean image-image similarity |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_scores` | per-anchor similarity values |

## Normalization

Current config:

- `lower = 0.0`
- `upper = 1.0`
- `higher_is_better = true`

## Official Rule

Always `diagnostic`.

## How To Read It

- High: the rollout stays visually close to the reference appearance.
- Low: the result diverges in identity, texture, lighting, or scene appearance.

This metric is useful for analysis, but it is intentionally kept out of headline reporting because close appearance copying is not the same as controllable generation quality.

## Common Misreads

- High score does not imply prompt correctness.
- High score may come from over-copying the reference rather than controlled generation.
- Low score is not automatically a failure if the benchmark intent allows broader variation.

## Implementation

- `worldarena/benchmark/metrics/alignment.py`
