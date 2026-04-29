# Anchor Consistency

`anchor_consistency` is a reference-relative drift probe. It asks whether prediction-reference appearance similarity stays stable from early anchors to late anchors.

## What It Measures

- change in reference-relative appearance similarity over time
- whether later anchors drift away from the reference compared with earlier anchors
- not self-consistency of the prediction alone, and not semantic correctness

## Detailed Algorithm

1. WorldArena pairs each prediction anchor frame with the reference anchor at the same protocol position.
2. Each pair is encoded by the same CLIP image encoder, then scored by embedding cosine similarity.
3. The pair score is `sim_i = clamp((cos_i + 1) / 2, 0, 1)`.
4. The raw metric is `raw = mean_i sim_i`.
5. The metric also records `late_minus_early = sim_last - sim_first`.
6. Current benchmark normalization is identity because the output is already in `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | prediction anchor frames |
| Reference input | paired reference anchor frames |
| Annotation input | none |
| Raw output | a scalar mean anchor similarity in `[0, 1]` |
| Normalized output | same scalar in `[0, 1]` |
| Details payload | `frame_scores`, `late_minus_early`, `frame_indices`, `sampled_frame_indices` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the CLIP backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | always `diagnostic` |
| Publication verdict | `Not ready` |

- This is a useful debugging signal for appearance drift, but it is not a benchmark-quality correctness metric.
- The score is still reference-relative, so it can reward copying and penalize acceptable controlled variation.
- Recommended use: read `late_minus_early` when diagnosing temporal appearance drift, not when ranking models for a paper.
