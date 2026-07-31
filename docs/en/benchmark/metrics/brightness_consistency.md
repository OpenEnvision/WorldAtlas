# Brightness Consistency

This metric quantifies the **temporal consistency of brightness distribution between each frame and the initial frame** in video sequences, addressing insufficient visual stability caused by inter-frame brightness shifts and abrupt changes. Unlike scalar metrics that compare only global mean luminance, Brightness Consistency preserves dark/mid/bright pixel ratios and is therefore more sensitive to changes in exposure distribution shape.

WorldArena uses this algorithm as the official metric **`brightness_distribution_consistency`**. The legacy scalar alias `brightness_consistency` remains diagnostic and is not part of the seven-metric official quality total.

### Three-Level Brightness Distribution Vector

Pixel grayscales are categorized into **dark**, **mid**, and **bright** levels to construct a 3-D brightness distribution vector:

\[
\mathbf{v}_t
=
\begin{pmatrix}
p_{\mathrm{dark}} \\
p_{\mathrm{mid}} \\
p_{\mathrm{bright}}
\end{pmatrix}
\]

where $p_{\mathrm{dark}}$ and $p_{\mathrm{bright}}$ are the fractions of pixels below the dark threshold and at or above the bright threshold, respectively, and $p_{\mathrm{mid}} = 1 - p_{\mathrm{dark}} - p_{\mathrm{bright}}$. WorldArena defaults: `dark_threshold=85`, `bright_threshold=170` (OpenCV gray scale $[0,255]$).

### Similarity and Modified Softmax

The distribution similarity of frame $t$ ($t \geq 2$) relative to the initial frame $\mathbf{v}_1$ uses the cosine inner product:

\[
S(a, b)
=
\frac{a \cdot b}{\|a\|\,\|b\|}
=
\frac{\sum_{i=1}^{n} a_i b_i}{
    \sqrt{\sum_{i=1}^{n} a_i^2}\,\sqrt{\sum_{i=1}^{n} b_i^2}
}
\tag{3}
\]

A modified Softmax transform enhances high-score segments for single-sample scores $x \in [0,1]$ ($\lambda > 0$):

\[
T(x)
=
\frac{e^{\lambda x} - 1}{e^{\lambda} - 1}
\tag{4}
\]

Default $\lambda = 4.0$ (`softmax_lambda`).

### Temporally Weighted Aggregation

For frame distance $d \in [1, T-1]$ ($d=1$ for the second frame), normalized exponentially decaying weights ($\alpha > 0$) are:

\[
w(d) = e^{-\alpha \cdot d},
\qquad
\hat{w}(d)
=
\frac{w(d)}{\sum_{k=1}^{T-1} w(k)}
\tag{5}
\]

Default $\alpha = 0.05$ (`temporal_decay`). The comprehensive brightness consistency score is:

\[
S_{\mathrm{Brightness}}
=
\sum_{d=1}^{T-1}
\hat{w}(d) \cdot T\!\left(S(\mathbf{v}_{d+1}, \mathbf{v}_1)\right)
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
      <img src="../../../assets/metric_examples/inputs/38c5df592fc3__image_static_photorealistic_aquatic_landscape_0098.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0098" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.9727</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0098</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pan right. An aquatic landscape with pebble beach, calm sea, and blue sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout keeps the bright coastal exposure stable; the lower-scoring rollout shifts toward a dark, murky distribution.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_aquatic_landscape_0098">
          <source src="../../../assets/metric_examples/6dcb1b7c126c__cosmos3-nano-image-static__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9991</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_aquatic_landscape_0098">
          <source src="../../../assets/metric_examples/curated_20260801/f1f662bdef18__image_static_photorealistic_aquatic_landscape_0098__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0264</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "WorldAtlas Arena — Brightness Consistency metric family"

    Three-bin brightness distribution vectors with cosine similarity to the initial frame, modified Softmax enhancement, and normalized exponential temporal weighting.
