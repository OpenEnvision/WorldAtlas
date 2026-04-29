# Motion Magnitude

`motion_magnitude` measures how much motion the prediction contains, regardless of whether that motion is correct.

## What It Measures

- overall motion amplitude in the prediction
- per-pair optical-flow strength
- not motion correctness and not smoothness

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. For each consecutive frame pair, the wrapped third-party metric runs `SEA-RAFT` to estimate optical flow.
3. It converts the flow into a magnitude map `m = sqrt(u^2 + v^2)`.
4. The pair statistic is the median value of that magnitude map over all pixels.
5. The raw metric is the mean pair statistic across all consecutive frame pairs.
6. WorldArena applies the current z-score calibration:
    - `avg = 3.2425`
    - `std = 3.4505`
    - `z_min = -0.8638`
    - `z_max = 2.7498`
    - output range `[0.25, 0.75]`

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none in the official backend |
| Annotation input | none |
| Raw output | one scalar average motion-magnitude score |
| Normalized output | one scalar in `[0, 1]` after z-score mapping |
| Details payload | `frame_count`, `native_score`, `frame_indices`, `sampled_frame_indices` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the flow backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- This metric is useful for distinguishing "too static" outputs from "more dynamic" outputs.
- It does not answer whether the motion is the *right* motion. A wildly wrong but energetic video can still score highly.
- Recommended use: keep it as a secondary dynamics metric, not as a headline model-quality result.
