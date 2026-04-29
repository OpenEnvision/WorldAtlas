# Background Consistency

`background_consistency` measures whether the prediction stays visually stable over time in a
background-like, whole-frame sense.

## What It Measures

- cross-frame stability of scene layout and background appearance
- temporal self-consistency inside the prediction
- a coarse proxy for whether the world behind the main motion flickers or changes unexpectedly

## Detailed Algorithm

1. WorldArena encodes sampled prediction frames with the CLIP image encoder
   `openai/clip-vit-base-patch16`.
2. For each frame `f_i` after the first one, it computes:
    - cosine similarity to the previous frame embedding
    - cosine similarity to the first frame embedding
3. Negative cosine values are clipped to zero.
4. The per-transition score is the mean of those two similarities.
5. The raw metric is the mean transition score across the sampled video.

Current integration intentionally keeps this metric simple. It does **not** segment the background and
therefore behaves as a whole-frame stability probe rather than a pure background-only metric.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | none |
| Annotation input | none |
| Raw output | one scalar self-consistency score in `[0, 1]` |
| Normalized output | bounded scalar after optional config normalization |
| Details payload | `transition_scores`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if CLIP is unavailable or fewer than two frames are available |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it is a cheap probe for temporal flicker and scene instability.
- Why it is limited: whole-frame CLIP similarity is only a rough background proxy.
- Recommended use: read it together with `subject_consistency` and `motion_smoothness`.
