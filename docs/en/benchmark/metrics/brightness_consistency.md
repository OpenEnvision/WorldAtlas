# Brightness Consistency

`brightness_consistency` measures temporal stability of global luminance in a generated video.

## What It Measures

- frame-to-frame brightness flicker
- global exposure stability over sampled prediction frames
- a low-level quality failure that can be missed by semantic or control metrics

## Detailed Algorithm

1. WorldArena reads the prediction's sampled video frames.
2. For each frame, it computes mean Rec. 709 luminance:
   \(Y = 0.2126R + 0.7152G + 0.0722B\).
3. It computes adjacent-frame absolute luminance deltas.
4. It also computes relative deltas using the larger neighboring luminance, with a small floor for dark clips.
5. Absolute and relative drift are each converted to a similarity with an exponential kernel.
6. The raw score is the geometric mean of the absolute-drift and relative-drift similarities.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | none |
| Annotation input | none |
| Raw output | scalar in `[0, 1]`; higher means more stable brightness |
| Normalized output | identity under the default config |
| Details payload | frame luminance, adjacent absolute/relative deltas, mean deltas, kernel settings |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if fewer than two frames are available |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it catches obvious exposure pumping and brightness flicker.
- Why it is limited: it does not know whether brightness changes are physically or semantically justified by a lighting transition.
- Recommended use: report as a supporting quality diagnostic next to `perceptual_quality`, `style_consistency`, and `sharpness_retention`.
