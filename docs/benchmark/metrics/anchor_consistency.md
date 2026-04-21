# Anchor Consistency

Measures whether reference-level appearance similarity is maintained from early to late anchors. It is a diagnostic drift probe, not an official metric.

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
- Like `reference_appearance_alignment`, it scores each prediction anchor against the paired reference anchor.
- It also exposes temporal drift via `details.late_minus_early`.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | mean anchor similarity across the rollout |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_scores` | per-anchor similarities |
| `details.late_minus_early` | last anchor minus first anchor similarity |

## Normalization

Current config:

- `lower = 0.0`
- `upper = 1.0`
- `higher_is_better = true`

## Official Rule

Always `diagnostic`.

## How To Read It

- High mean score: the rollout stays close to reference appearance over time.
- Negative `late_minus_early`: late anchors drift away from the reference relative to early anchors.
- Positive `late_minus_early`: appearance similarity is stronger later than earlier.

## Common Misreads

- This is not self-consistency between predicted anchors only; it is measured relative to the reference anchors.
- It is a drift signal, not a semantic metric.

## Implementation

- `worldarena/benchmark/metrics/alignment.py`
