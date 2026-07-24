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
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `brightness_distribution_consistency`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/38c5df592fc3__image_static_photorealistic_aquatic_landscape_0098.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0098" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.9991</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0098</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. An aquatic landscape with pebble beach, calm sea, and blue sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_aquatic_landscape_0098">
      <source src="../../../assets/metric_examples/6dcb1b7c126c__cosmos3-nano-image-static__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9991</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_aquatic_landscape_0098">
      <source src="../../../assets/metric_examples/a153996e000e__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/70502be4e8e4__image_static_photorealistic_workspaces_0012.png" alt="Input image: image_static_photorealistic_workspaces_0012" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.9990</div>
      <h4><code>image_static_photorealistic_workspaces_0012</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. A minimalist workspace with desk, laptop, and chair. Camera move right. A minimalist workspace with desk, laptop, and chair. Camera pull out. A minimalist workspace with desk, laptop, and chair.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_workspaces_0012">
      <source src="../../../assets/metric_examples/4ce7235be4d7__yume_sample5b__image_static_photorealistic_workspaces_0012.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9999</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_workspaces_0012">
      <source src="../../../assets/metric_examples/940042a83949__dreamx_world__image_static_photorealistic_workspaces_0012.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0009</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2a2ecb6a511d__image_static_photorealistic_passageways_0050.jpg" alt="Input image: image_static_photorealistic_passageways_0050" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 0.9985</div>
      <h4><code>image_static_photorealistic_passageways_0050</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move right. A modern walk-in closet with shelving, shoe rack, and mirror.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_passageways_0050">
      <source src="../../../assets/metric_examples/9d5b5b64512b__hy_worldplay_ar_rl_image_test_balanced_5000_5s__image_static_photorealistic_passageways_0050.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_passageways_0050">
      <source src="../../../assets/metric_examples/a0c6cd5ddefe__lingbot_world__image_static_photorealistic_passageways_0050.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0015</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/573103498bf2__image_static_photorealistic_workspaces_0086.png" alt="Input image: image_static_photorealistic_workspaces_0086" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 0.9983</div>
      <h4><code>image_static_photorealistic_workspaces_0086</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move left. An empty office with glass partition, windows, and carpeted floor.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_workspaces_0086">
      <source src="../../../assets/metric_examples/755a7634ea65__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_workspaces_0086.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9999</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_workspaces_0086">
      <source src="../../../assets/metric_examples/d5997ae5d656__dreamx_world__image_static_photorealistic_workspaces_0086.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0016</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/47bdf207f40a__image_static_stylized_terrestrial_landscape_0138.png" alt="Input image: image_static_stylized_terrestrial_landscape_0138" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 0.9968</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0138</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pull out. An expansive desert with mountains, path, and rocks.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_terrestrial_landscape_0138">
      <source src="../../../assets/metric_examples/d85d8c9f7579__hy_worldplay_ar_rl_image_test_balanced_5000_5s__image_static_stylized_terrestrial_landscape_0138.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_terrestrial_landscape_0138">
      <source src="../../../assets/metric_examples/b2a98165e36f__cosmos3-nano-image-static-8gpu-singleload-20260612_170108__image_static_stylized_terrestrial_landscape_0138.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0032</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/05921dcab542__image_static_photorealistic_aquatic_landscape_0162.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0162" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 0.9964</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0162</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move right. An aquatic landscape with lake, shoreline, and trees.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_aquatic_landscape_0162">
      <source src="../../../assets/metric_examples/9fa9ba3b8c50__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_aquatic_landscape_0162.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_aquatic_landscape_0162">
      <source src="../../../assets/metric_examples/ea022fe1d259__hy_worldplay_ar_image_test_balanced_5000_5s__image_static_photorealistic_aquatic_landscape_0162.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0036</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "WorldAtlas Arena — Brightness Consistency metric family"

    Three-bin brightness distribution vectors with cosine similarity to the initial frame, modified Softmax enhancement, and normalized exponential temporal weighting.
