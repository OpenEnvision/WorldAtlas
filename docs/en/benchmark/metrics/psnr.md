# PSNR

`psnr` measures frame-wise signal fidelity between prediction and reference.

## What It Measures

- pixel-level similarity to the reference
- reconstruction fidelity, not semantic correctness
- whether the generated frames stay numerically close to the reference frames

## Detailed Algorithm

1. WorldArena pairs sampled prediction and reference frames at aligned time indices.
2. The prediction frame is resized to the reference frame resolution if needed.
3. For each frame pair, WorldArena computes
  `PSNR = 20 * log10(255) - 10 * log10(MSE)`.
4. Perfect matches are clamped to `100.0` instead of infinity.
5. The raw metric is the mean frame PSNR.

## Inputs And Outputs


| Item              | Current implementation                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Prediction input  | sampled prediction frames                                                                         |
| Reference input   | sampled reference frames                                                                          |
| Annotation input  | none                                                                                              |
| Raw output        | one scalar mean PSNR value                                                                        |
| Normalized output | min/max scaled diagnostic score from `config/benchmark.yaml`                                      |
| Details payload   | `frame_scores`, frame index metadata                                                              |
| Failure mode      | returns `raw = null`, `normalized = null`, and an error string if no aligned frames are available |


## Integration Readiness


| Item                | Verdict           |
| ------------------- | ----------------- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |


- Why it is useful: it is simple, stable, and easy to interpret.
- Why it is limited: it strongly rewards copying the reference and says little about semantic validity.
- Recommended use: keep it as a reconstruction-style appendix metric only.