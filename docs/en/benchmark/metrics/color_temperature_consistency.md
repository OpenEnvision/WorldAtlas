# Color Temperature Consistency

`color_temperature_consistency` measures temporal drift in the prediction's global hue balance.

## What It Measures

- color-cast instability across a rollout
- global hue drift that can make a video look inconsistent
- an HSV-space proxy for color-temperature consistency, not a calibrated Kelvin measurement

## Detailed Algorithm

1. WorldArena reads the prediction's sampled video frames.
2. Each frame is converted from RGB to HSV.
3. The metric estimates one global frame hue by averaging hue vectors on the unit circle.
4. Pixel weights are `saturation ** saturation_power * value`, so saturated visible regions dominate the estimate.
5. Adjacent-frame hue distances are computed as circular distances in `[0, 1]`.
6. The weighted mean hue drift is converted to a score with an exponential kernel.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | none |
| Annotation input | none |
| Raw output | scalar in `[0, 1]`; higher means less hue drift |
| Normalized output | identity under the default config |
| Details payload | frame hue angles, hue reliability weights, pairwise hue distances, kernel settings |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if fewer than two frames or no sufficiently saturated frame pairs are available |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it catches frame-to-frame color-temperature jumps and color-cast drift.
- Why it is limited: HSV hue is a proxy; it is not a color-calibrated white-point or Kelvin estimator.
- Recommended use: keep it as a low-level visual stability diagnostic, not as a standalone color-science claim.
