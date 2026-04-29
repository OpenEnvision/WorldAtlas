# Physical Motion Fidelity

`pmf` measures whether the prediction preserves the same broad motion dynamics as the reference rollout, while being deliberately less sensitive to absolute spatial position and global brightness.

WorldArena currently implements PMF directly in-repo with a NumPy FFT backend.

## What It Measures

- similarity of spatiotemporal motion-energy patterns between prediction and reference
- robustness to simple translation-like offsets and brightness scaling
- not prompt following, not object identity, and not pixel-perfect reconstruction

## Detailed Algorithm

1. WorldArena reads the rollout evaluation window from the prediction and reference videos.
2. If the two videos expose different numbers of sampled frames, it uniformly resamples both to the shorter length.
3. If the resolutions differ, it resizes prediction frames to the reference resolution.
4. It stacks each video as an RGB tensor with shape `[T, H, W, C]`.
5. It applies a 3D discrete Fourier transform over `H x W x T` independently for each RGB channel.
6. It converts the Fourier coefficients into a summed power spectrum:
    - `power(u, v, s) = sum_c |F_c(u, v, s)|^2`
7. It normalizes that power spectrum into a probability-like energy distribution:
    - `E(u, v, s) = power(u, v, s) / sum power`
8. It computes total variation distance between prediction and reference energy distributions:
    - `tv = 0.5 * sum |E_pred - E_ref|`
9. It reports:
    - `pmf = -log(max(tv, 1e-12))`
10. WorldArena then applies bounded normalization with:
    - `lower = 0.0`
    - `upper = 6.0`
    - higher is better

Higher PMF means the two videos have more similar motion-energy structure. Perfect spectral agreement saturates at the epsilon floor rather than returning mathematical infinity.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | rollout-window prediction video |
| Reference input | rollout-window reference video |
| Annotation input | none |
| Raw output | one scalar PMF score |
| Normalized output | one scalar in `[0, 1]` after direct bounded scaling |
| Details payload | `tv_distance`, `epsilon_floor`, `aligned_frame_count`, `reference_resolution`, `prediction_frame_count`, `reference_frame_count`, `temporal_alignment`, `prediction_resized`, `reference_resized`, plus frame-index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if either video has fewer than two frames or decoding/alignment fails |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` for `image_static`, `image_dynamic`, and `video_dynamic` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- This metric is substantially better than plain motion-amplitude probes when you care about whether the generated rollout follows the same broad physical dynamics as the reference.
- Pure image references usually produce no score because PMF needs at least two reference frames.
- It is still not a complete correctness metric. Two videos can share similar motion spectra while differing in object identity, causal interactions, or semantic correctness.
- Because it is reference-paired, it is best used together with control, persistence, and quality metrics rather than as a single-number claim about world modeling.
