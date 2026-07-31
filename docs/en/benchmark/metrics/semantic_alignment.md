# Semantic Alignment

Semantic Alignment measures whether a generated video preserves scene-level content described in annotation sidecars, independent of the benchmark prompt \(T\). It complements Prompt Alignment by scoring agreement with caption-derived reference text rather than the explicit generation instruction.

WorldArena constructs reference text \(R\) from `caption.json` under `sample.annotation_path`, optionally appending camera-instruction labels. When annotation text is unavailable, \(R\) falls back to `sample.prompt_current` or `sample.prompt_target`. For decoded frame \(f_i\), CLIP ViT-B/16 yields L2-normalized image embedding \(\mathbf{e}_i\) and text embedding \(\mathbf{r} = \phi_T(R)\). The per-frame score is

\[
\tilde{s}_i = \mathrm{clamp}\!\left(\frac{\cos(\mathbf{e}_i, \mathbf{r}) + 1}{2},\, 0,\, 1\right)
\tag{1}
\]

The raw metric aggregates over all evaluated frames:

\[
S_{\mathrm{sem}} = \frac{1}{N}\sum_{i=1}^{N} \tilde{s}_i
\tag{2}
\]

Higher \(S_{\mathrm{sem}}\) indicates closer semantic match to the annotated reference description. The metric does not substitute for instruction-following scores or object-level verification.

<!-- metric-examples:start -->
## No Verified Same-Case Pair Yet

> The previous example came from readiness/smoke reports outside `artifacts/reports`. It has been removed from the formal-results gallery until a verified same-case pair exists in the retained report snapshot.
<!-- metric-examples:end -->

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
