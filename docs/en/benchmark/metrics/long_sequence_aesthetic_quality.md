# Long Sequence Aesthetic Quality

**long_sequence_aesthetic_quality** measures **mean no-reference aesthetic appeal** over prediction frames. Each frame $I_t$ is encoded by a frozen CLIP ViT-L/14 backbone; a LAION-Aesthetics linear head produces per-frame scores normalized to $[0,1]$.

\[
S_{\mathrm{aes}} = \frac{1}{T}\sum_{t=1}^{T} s_t^{\mathrm{aes}}
\tag{1}
\]

Higher is better. Long-rollout quality signal computed prediction-only via the `long_sequence` backend (`aesthetic_quality` dimension). See [Long-Sequence Metrics](long_sequence.md) for the shared runtime.

<!-- metric-examples:start -->
## No Verified Same-Case Pair Yet

> The retained candidates do not contain a two-sided input-verified pair for this metric. The former six-card gallery has been removed.
<!-- metric-examples:end -->

## References {#reference}

!!! quote "LAION-Aesthetics Predictor (CLIP-based linear aesthetic regressor)"

    Used via pyiqa `laion_aes`; CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings.
