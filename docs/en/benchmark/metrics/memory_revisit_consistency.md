# Memory Revisit Consistency

`memory_revisit_consistency` evaluates 360-degree revisit rollouts: the model should move away from the starting view and return to the same generated scene.

## What It Measures

- whether the first and last frames close the visual loop
- whether intermediate frames show enough progress to avoid rewarding a static video
- a diagnostic signal for orbit, revisit, and loop-closing samples

## Current Algorithm

1. The metric reads protocol-sampled prediction frames.
2. It compares the first frame with the resized last frame.
3. DINOv2 features are extracted for all sampled frames.
4. Closure uses three components:
    - `dino_revisit = sim_dino(start, end)`
    - `ssim_revisit = SSIM(start, end)`
    - `mse_revisit = 1 - clamp(MSE(start, end) / mse_tau, 0, 1)`, default `mse_tau = 0.08`
5. The default closure weights are `dino = 0.55`, `ssim = 0.25`, `mse = 0.20`.
6. Intermediate progress is measured at the 25%, 50%, and 75% frames using DINO displacement from the first frame.
7. `progress_score` maps mean displacement through the default interval `[0.03, 0.18]`.
8. Final raw score is `closure_score * ((1 - gate) + gate * progress_score)`, default `gate = 0.35`.

A fully static video with perfect first-last closure is capped around `0.65`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | protocol-sampled prediction video frames |
| Reference input | unused |
| Annotation input | unused |
| Backend | `dinov2` |
| Raw output | `[0, 1]` revisit consistency score |
| Normalized output | identity mapping by default |
| Detail fields | `dino_revisit`, `ssim_revisit`, `mse`, `mse_revisit`, `closure_score`, `quarter_indices`, `quarter_dino_similarities`, `mean_quarter_displacement`, `progress_score`, `progress_gate_weight` |

## Aggregation

| Item | Status |
| --- | --- |
| Suite | `experimental` |
| Aggregation | always `diagnostic` |
| Dimension | `long_sequence` |

Key limit: the metric does not verify true 360-degree camera geometry. It only checks visual departure and return.
