# Subject Consistency

`subject_consistency` measures whether the main visual subject remains stable across prediction
frames.

## What It Measures

- cross-frame stability of subject identity inside the prediction
- temporal self-consistency, not correctness against a reference
- whether the generated subject drifts, mutates, or swaps identity over time

## Detailed Algorithm

1. WorldArena encodes sampled prediction frames with `facebook/dinov2-base`.
2. For each frame `f_i` after the first one, it computes:
    - cosine similarity to the previous frame embedding
    - cosine similarity to the first frame embedding
3. Negative cosine values are clipped to zero.
4. The per-transition score is the mean of those two similarities.
5. The raw metric is the mean transition score across the sampled video.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | none |
| Annotation input | none |
| Raw output | one scalar self-consistency score in `[0, 1]` |
| Normalized output | bounded scalar after optional config normalization |
| Details payload | `transition_scores`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if DINOv2 is unavailable or fewer than two frames are available |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it catches identity drift that prompt metrics and quality metrics can miss.
- Why it is limited: it has no explicit subject mask or identity labels, so it remains a self-proxy.
- Recommended use: keep it as a temporal stability diagnostic, not as a headline correctness metric.
