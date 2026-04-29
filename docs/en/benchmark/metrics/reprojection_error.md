# Reprojection Error

`reprojection_error` measures geometric self-consistency of the prediction under the wrapped DROID-SLAM reconstruction path.

## What It Measures

- how consistently the prediction can be reconstructed and reprojected by the SLAM backend
- prediction-only geometric consistency
- not prompt alignment and not camera-ground-truth accuracy

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. The wrapped third-party metric runs `DROID-SLAM` on those frames to estimate a trajectory and internal geometry.
3. At termination, the DROID backend returns `valid_errors`, which are the per-observation reprojection errors accepted by that runtime.
4. The raw metric is `raw = mean(valid_errors)`.
5. WorldArena normalizes the scalar with empirical bounds:
    - `empirical_min = 0.0`
    - `empirical_max = 1.0719`
    - lower is better
6. So the normalized score is effectively `1 - clamp((raw - 0.0) / (1.0719 - 0.0), 0, 1)`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | one scalar reprojection error |
| Normalized output | one scalar in `[0, 1]`, where higher is better |
| Details payload | `frame_count`, `native_score` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if SLAM reconstruction fails or no score is produced |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- This is a useful geometry-consistency signal, but it is still a prediction-only SLAM score. It is not geometry ground truth.
- The wrapped third-party path uses the same fixed-intrinsics SLAM assumptions as the current camera metric, so it should be interpreted cautiously across heterogeneous data.
- Recommended use: acceptable as a supporting geometric-consistency metric, not as a headline correctness result.
