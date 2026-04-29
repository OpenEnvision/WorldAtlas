# Memory Revisit Metric Design

Task: generate a 60s+ in-place 360-degree rotation from one image, then return to the starting view.

Primary metric implemented as `memory_revisit_consistency`:

```text
I0 = first generated frame
I1 = last generated frame
Q  = generated frames at 25%, 50%, 75%

dino_revisit = (cos(DINO(I0), DINO(I1)) + 1) / 2
ssim_revisit = SSIM(I0, I1)
mse_revisit  = 1 - clamp01(MSE(I0/255, I1/255) / 0.08)

closure = 0.55 * dino_revisit + 0.25 * ssim_revisit + 0.20 * mse_revisit

quarter_displacement = mean(1 - dino_similarity(I0, q) for q in Q)
progress_score = clamp01((quarter_displacement - 0.03) / (0.18 - 0.03))

memory_revisit_consistency = closure * (0.65 + 0.35 * progress_score)
```

Why this shape:

- DINO is the primary signal because start/end should preserve semantic layout and landmarks even if the generated frame is not pixel-perfect.
- SSIM and MSE keep pressure on geometry, color, and local appearance alignment when the camera returns to the same yaw.
- Quarter-turn progress prevents a static video from receiving a perfect memory score. A model that never rotates can still have high closure, but its final score is capped by the progress gate.
- The metric compares generated first vs generated last frame, not input vs output, so it measures model memory after rollout rather than first-frame reconstruction quality.

Recommended reporting:

- Use `memory_revisit_consistency` as the aggregate score.
- Also report `dino_revisit`, `ssim_revisit`, `mse_revisit`, and `progress_score`.
- Treat cases with high closure but low progress as action-following failures, not true memory success.
