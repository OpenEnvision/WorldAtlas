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
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/bfa2ac37072c__image_static_photorealistic_terrestrial_landscape_0003__input.jpg" alt="Input image: image_static_photorealistic_terrestrial_landscape_0003" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8218</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0003</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit right. A mountain landscape with mountains, trees, and mist.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_terrestrial_landscape_0003">
          <source src="../../../assets/metric_examples/curated_20260801/c7092b92b672__image_static_photorealistic_terrestrial_landscape_0003__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9839</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_terrestrial_landscape_0003">
          <source src="../../../assets/metric_examples/curated_20260801/7a5ad1b24e1b__image_static_photorealistic_terrestrial_landscape_0003__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.1621</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d43de2a66deb__image_static_stylized_suburb_0035__input.jpg" alt="Input image: image_static_stylized_suburb_0035" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9079</div>
      <h4><code>image_static_stylized_suburb_0035</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit right. A stylized cityscape with tower, skyscrapers, and trees.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_suburb_0035">
          <source src="../../../assets/metric_examples/curated_20260801/83b4220004c5__image_static_stylized_suburb_0035__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9988</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_suburb_0035">
          <source src="../../../assets/metric_examples/curated_20260801/088f33146a5d__image_static_stylized_suburb_0035__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0909</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/a2fd68c1863b__image_static_stylized_passageways_0041__input.jpg" alt="Input image: image_static_stylized_passageways_0041" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9815</div>
      <h4><code>image_static_stylized_passageways_0041</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. A stylized courtyard with arches, columns, and round table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_passageways_0041">
          <source src="../../../assets/metric_examples/curated_20260801/a1b55065b899__image_static_stylized_passageways_0041__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_passageways_0041">
          <source src="../../../assets/metric_examples/5063767b647a__astra__image_static_stylized_passageways_0041.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0185</div>
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
