# Depth Accuracy

Depth Accuracy is a paired metric comparing relative monocular depth structure between prediction and reference anchor frames. Both sides pass through the same Depth Anything V2 model; no ground-truth depth is used.

For pair \(i\), let \(D_i^{\mathrm{pred}}\) and \(D_i^{\mathrm{ref}}\) be predicted depth maps. Median-ratio alignment yields

\[
\alpha_i = \frac{\mathrm{median}(D_i^{\mathrm{ref}})}{\max(\mathrm{median}(D_i^{\mathrm{pred}}), 10^{-6})},
\qquad
\tilde{D}_i^{\mathrm{pred}} = \alpha_i D_i^{\mathrm{pred}}
\tag{1}
\]

Over valid pixels \(\Omega_i = \{(x,y) \mid D_i^{\mathrm{ref}}(x,y) > 10^{-3}\}\), frame-level absolute relative error is

\[
e_i = \frac{1}{|\Omega_i|}\sum_{(x,y)\in\Omega_i} \frac{\left|\tilde{D}_i^{\mathrm{pred}}(x,y) - D_i^{\mathrm{ref}}(x,y)\right|}{\max(D_i^{\mathrm{ref}}(x,y), 10^{-6})}
\tag{2}
\]

The raw score \(e_{\mathrm{depth}} = \frac{1}{N}\sum_i e_i\) averages over \(N\) aligned pairs. Lower error maps to a higher normalized score. The metric evaluates coarse front–back ordering preservation, not metric-depth ground truth or camera trajectory accuracy.

<!-- metric-examples:start -->
## No Verified Same-Case Pair Yet

> The retained formal reports currently provide only one prediction run for this metric, so no verified same-case high/low output pair is available. Different-case extrema are intentionally not shown.
<!-- metric-examples:end -->

## References {#reference}

!!! quote "Depth Anything V2"

    Yang et al., 2024.
    [https://arxiv.org/abs/2406.09414](https://arxiv.org/abs/2406.09414)
