# SSIM

`ssim` measures frame-wise structural similarity between prediction and reference.

## What It Measures

- structural closeness to the reference
- luminance, contrast, and local-structure agreement
- reconstruction-style similarity with a mild perceptual bias

## Detailed Algorithm

1. WorldArena pairs sampled prediction and reference frames at aligned time indices.
2. The prediction frame is resized to the reference frame resolution if needed.
3. For each RGB channel, it computes a standard Gaussian-window SSIM map.
4. Channel SSIM scores are averaged into one frame score.
5. The raw metric is the mean frame SSIM.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | sampled reference frames |
| Annotation input | none |
| Raw output | one scalar mean SSIM score in `[0, 1]` |
| Normalized output | bounded scalar under the default config |
| Details payload | `frame_scores`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if no aligned frames are available |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it is more structure-aware than raw PSNR.
- Why it is limited: like PSNR, it still rewards reference copying and should not stand in for semantic
  or control correctness.
- Recommended use: report it only as a supporting, full-reference diagnostic.
