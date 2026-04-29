# Motion Smoothness

`motion_smoothness` measures how easy it is to interpolate the prediction temporally.

## What It Measures

- temporal stability and interpolatability
- flicker and abrupt transition severity
- not motion strength and not semantic correctness

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. The wrapped third-party metric splits the decoded frames into:
    - even frames: `I_0, I_2, I_4, ...`
    - odd frames: `I_1, I_3, I_5, ...`
3. For each pair of neighboring even frames `(I_{2k}, I_{2k+2})`, `VFIMamba` predicts the middle frame `Î_{2k+1}`.
4. The predicted middle frame is compared with the actual odd frame `I_{2k+1}` using three component metrics:
    - `MSE`
    - `SSIM`
    - `LPIPS`
5. The raw metric is a three-component average across all interpolation windows:
    - `raw.mse = mean window MSE`
    - `raw.ssim = mean window SSIM`
    - `raw.lpips = mean window LPIPS`
6. WorldArena normalizes each component independently:
    - `mse` in `[0.0, 82.4014]`, lower is better
    - `ssim` in `[0.9224, 1.0]`, higher is better
    - `lpips` in `[0.0, 0.0228]`, lower is better
7. The final normalized score is the arithmetic mean of the three normalized components.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | `{"mse": float, "ssim": float, "lpips": float}` |
| Normalized output | one scalar in `[0, 1]` after component normalization and arithmetic mean reduction |
| Details payload | `frame_count`, `native_components`, `frame_indices`, `sampled_frame_indices` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the interpolation runtime fails |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- This is a legitimate temporal-stability signal, but it does not tell you whether the motion is semantically correct.
- A nearly static video can score well here while still being a bad generative result.
- Recommended use: keep it as a secondary temporal-quality metric, ideally next to motion magnitude or semantic metrics that test different failure modes.
