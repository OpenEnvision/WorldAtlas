# Optical Flow AEPE

`optical_flow_aepe` measures forward-backward optical-flow self-consistency on the prediction.

## What It Measures

- short-range temporal consistency between consecutive frames
- whether forward and backward optical flow agree with each other
- not motion ground truth and not target motion correctness

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. For each consecutive frame pair `(I_t, I_{t+1})`, the wrapped third-party metric runs `SEA-RAFT` twice:
    - forward flow `f_{t->t+1}`
    - backward flow `f_{t+1->t}`
3. Both flows are cropped to a central region using `crop = 30`.
4. For each pixel `x` in that crop, the metric applies a round trip:
    - move forward with `f_{t->t+1}(x)`
    - sample the backward flow at the rounded destination
    - compare the warped-back position with the original `x`
5. The pair score is the mean endpoint error of that round trip over all cropped pixels.
6. The raw metric is the mean pair score across all consecutive frame pairs.
7. WorldArena normalizes the scalar with `empirical_min = 0.0`, `empirical_max = 1.192`, and `higher_is_better = false`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | one scalar forward-backward consistency error |
| Normalized output | one scalar in `[0, 1]`, where higher is better |
| Details payload | `frame_count`, `native_score` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the SEA-RAFT runtime is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- Despite the name, this is **not** classic AEPE against ground-truth flow. It is a forward-backward self-consistency error.
- That makes it useful for temporal artifact analysis, but not sufficient for claiming that the model produced the correct motion.
- Recommended use: keep it as a supporting consistency metric only.
