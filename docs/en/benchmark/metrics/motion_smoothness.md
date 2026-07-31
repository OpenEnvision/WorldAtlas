# Motion Smoothness

Motion Smoothness measures **temporal interpolatability** of a generated video under a frame-interpolation prior. VFIMamba reconstructs discarded odd frames from neighboring even frames; low reconstruction error indicates smooth, continuous motion—not correct semantics or magnitude.

For window \(k\), \(\hat{I}_{2k+1} = \Phi(I_{2k}, I_{2k+2})\) is compared to ground-truth odd frame \(I_{2k+1}\) via MSE, SSIM, and LPIPS. Raw components average over \(K\) windows:

\[
E_{\mathrm{mse}} = \frac{1}{K}\sum_k \mathrm{MSE}(\hat{I}_{2k+1}, I_{2k+1}),\quad
S_{\mathrm{ssim}} = \frac{1}{K}\sum_k \mathrm{SSIM}(\hat{I}_{2k+1}, I_{2k+1}),\quad
E_{\mathrm{lpips}} = \frac{1}{K}\sum_k \mathrm{LPIPS}(\hat{I}_{2k+1}, I_{2k+1})
\tag{1}
\]

Decoded `uint8` frames are converted to `float32` before MSE subtraction and squaring, so negative pixel differences cannot wrap modulo 256.

Each component is normalized independently; the final score is their arithmetic mean. Higher \(S_{\mathrm{smooth}}\) indicates smoother generated motion. The metric is **official** when computation succeeds and requires no reference video.

The configured MSE empirical bound predates `v2_float32_mse`. Corrected raw reports are versioned, but their normalized score remains provisional until that bound is recalibrated on the corrected benchmark corpus.

<!-- metric-examples:start -->
## Example Pending Regeneration

The previous comparison used scores produced before `v2_float32_mse` and is intentionally withheld. The reviewed video assets and historical audit manifest are retained; candidate selection and displayed scores must be regenerated from corrected reports.
<!-- metric-examples:end -->

## References {#reference}

!!! quote "VFIMamba: Video Frame Interpolation with State Space Models"

    Zhang et al., NeurIPS 2024.

!!! quote "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Zhang et al., CVPR 2018.
