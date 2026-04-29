# Motion Accuracy

`motion_accuracy` is intended to measure motion behavior inside supervised dynamic regions, but the current implementation is narrower than that name suggests.

## What It Measures

- whether tracked foreground regions move more strongly than the background
- mask-conditioned motion salience
- not a strict comparison against a target trajectory, target direction, or target velocity profile

## Detailed Algorithm

1. WorldArena requires a `mask_dir` and loads every image file inside it as the initial object masks.
2. The wrapped third-party metric builds a `SAM2` video predictor and uses the mask files as first-frame supervision only.
3. `SAM2` propagates those masks through the video. The current implementation then dilates every propagated mask by `10` pixels.
4. For each consecutive frame pair `(I_t, I_{t+1})`, the metric runs `SEA-RAFT` to estimate optical flow.
5. It converts flow into a magnitude map `m_t = sqrt(u^2 + v^2)`.
6. Using the propagated mask for that timestep, it computes:
    - `object_max = max(m_t inside mask)`
    - `background_max = max(m_t outside mask)`
7. The pair score is `pair_score = object_max - background_max`.
8. The raw metric is the mean pair score across all consecutive frame pairs.
9. WorldArena applies the current z-score calibration:
    - `avg = -0.1965`
    - `std = 2.3687`
    - `z_min = -1.6147`
    - `z_max = 1.5180`
    - output range `[0.25, 0.75]`

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | first-frame mask directory |
| Raw output | one scalar mean foreground-minus-background motion score |
| Normalized output | one scalar in `[0, 1]` after z-score mapping |
| Details payload | `frame_count`, `mask_count`, `mask_dir`, `generate_type`, runtime device details, `native_score` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if masks are unavailable or the heavy runtime fails |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when masks exist and a numeric score is produced |
| Publication verdict | `Not ready` |

- The current score is not a strict motion-faithfulness metric. It measures whether the masked region has a larger **maximum** flow magnitude than the background, which is sensitive to outliers and does not verify the desired motion pattern.
- The current `t2v` branch is incomplete. The third-party metric loads `Grounding DINO` for `t2v`, but the wrapped WorldArena path passes `objects_list = []`, and the metric's scoring code does not actually use those objects during `_compute_scores()`.
- Because the metric is based on mask propagation from the first frame, any `SAM2` failure directly contaminates the score.
- Recommended use: do not use the current implementation as a paper-grade motion-control result. Keep it experimental until the `t2v` path and the score definition are tightened and validated.
