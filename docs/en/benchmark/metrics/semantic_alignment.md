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
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `semantic_alignment`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/fe16fdec2c52__image_dynamic_photorealistic_articulated_0001.jpg" alt="Input image: image_dynamic_photorealistic_articulated_0001" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.0678</div>
      <h4><code>image_dynamic_photorealistic_articulated_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. A bird of prey glides forward low over a grassy field, wings spread wide with subtle body and wing adjustments. The wooden fence post, wild grasses, and vines remain still.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: _readiness_echo_dynamic_smoke_after image_dynamic_photorealistic_articulated_0001">
      <source src="../../../assets/metric_examples/8f778faac6c6___readiness_echo_dynamic_smoke_after__image_dynamic_photorealistic_articulated_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.6668</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: stable-virtual-camera-image-dynamic-5s image_dynamic_photorealistic_articulated_0001">
      <source src="../../../assets/metric_examples/73ddbcad1071__stable-virtual-camera-image-dynamic-5s__image_dynamic_photorealistic_articulated_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.5990</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
