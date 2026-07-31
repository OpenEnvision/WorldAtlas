# Style Consistency
This metric evaluates the stability of visual style—color palette, texture, and coarse appearance statistics—across scenes within a generated rollout.  Given protocol-selected anchor frames $\{I_t\}_{t=1}^{T}$ with $T \geq 2$, the first frame $I_1$ serves as the style reference; each later anchor $I_{d+1}$ ($d \in [1, T-1]$) is compared against $I_1$ via multi-layer VGG19 Gram representations.

Each frame $I_t$ is resized ($512 \times 512$ on CUDA, $128 \times 128$ on CPU), converted to a tensor, and normalized with ImageNet mean and standard deviation before a forward pass through VGG19 `features`. At the outputs of convolutional layers $\mathrm{conv}_1, \ldots, \mathrm{conv}_5$ (before ReLU), feature maps $F_{t,l} \in \mathbb{R}^{C \times H \times W}$ are extracted. The Gram matrix is defined as

\[
G(F)
=
\frac{
    \mathrm{vec}(F)\,\mathrm{vec}(F)^{\top}
}{
    C \cdot H \cdot W
}
\tag{1}
\]

where $\mathrm{vec}(\cdot)$ denotes vectorization. The layer-wise style distance for frame $d+1$ at layer $l$ is

\[
\delta_{d,l}
=
\mathrm{MSE}\!\left(
    G(F_{d+1,l}),\,
    G(F_{1,l})
\right)
\tag{2}
\]

The per-frame distance is the sum over five layers:

\[
d_d
=
\sum_{l \in \{\mathrm{conv}_1, \ldots, \mathrm{conv}_5\}}
\delta_{d,l}
\tag{3}
\]

The raw score is the arithmetic mean over later anchors (**lower is better**):

\[
\mathrm{raw}
=
\frac{1}{T-1}
\sum_{d=1}^{T-1} d_d
\tag{4}
\]

Normalization follows configured empirical bounds $\varepsilon_{\min}=0$ and $\varepsilon_{\max}=0.007$ (`higher_is_better=false`). The raw value is first clamped to the empirical range, then mapped to a **higher-is-better** score in $[0,1]$:

\[
\mathrm{raw}'
=
\mathrm{clip}(\mathrm{raw},\, \varepsilon_{\min},\, \varepsilon_{\max})
\tag{5}
\]

\[
S_{\mathrm{style}}
=
1
-
\frac{
    \mathrm{raw}' - \varepsilon_{\min}
}{
    \varepsilon_{\max} - \varepsilon_{\min}
}
\tag{6}
\]

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/f7d43b5c2adb__image_static_photorealistic_passageways_0144__input.jpg" alt="Input image: image_static_photorealistic_passageways_0144" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.0103</div>
      <h4><code>image_static_photorealistic_passageways_0144</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pan left. A modern hallway with geometric walls, adjacent corridor, and metallic ceiling lights.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout stays within a coherent photoreal geometric-interior style; the lower-scoring rollout switches to an unrelated reflective striped appearance.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0144">
          <source src="../../../assets/metric_examples/curated_20260801/e33a33498b58__image_static_photorealistic_passageways_0144__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9999</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>style_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0144">
          <source src="../../../assets/metric_examples/curated_20260801/a5ac53bfecb4__image_static_photorealistic_passageways_0144__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.9896</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>style_consistency</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "Very Deep Convolutional Networks for Large-Scale Image Recognition"

    Karen Simonyan and Andrew Zisserman.
    arXiv:1409.1556, 2014.
    [https://arxiv.org/pdf/1409.1556](https://arxiv.org/pdf/1409.1556)
