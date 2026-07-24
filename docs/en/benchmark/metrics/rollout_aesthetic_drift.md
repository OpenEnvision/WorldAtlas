# Rollout Aesthetic Drift

**rollout_aesthetic_drift** measures **temporal aesthetic degradation** along a rollout. Per-frame LAION aesthetic scores are split into $N=10$ equal segments; drift is the relative drop from the best to the worst segment mean.

\[
D_{\mathrm{aes}} = \frac{\bar{s}_{\mathrm{best}} - \bar{s}_{\mathrm{worst}}}{|\bar{s}_{\mathrm{best}}|}
\tag{1}
\]

Lower is better. The value is clamped to $[0,1]$. Computed prediction-only. Backend: `quality_drift` (aesthetic component via pyiqa `laion_aes`).

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `rollout_aesthetic_drift`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2c2b9c480799__image_static_photorealistic_terrestrial_landscape_0100.jpg" alt="Input image: image_static_photorealistic_terrestrial_landscape_0100" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.5320</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0100</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. A mountain landscape with valley, river, and mountain peaks. Camera pull out. A mountain landscape with forest, wildlife, and deer. Camera move right. A mountain landscape with village, smoke, and chimneys.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_terrestrial_landscape_0100">
      <source src="../../../assets/metric_examples/3ef2096fbf6e__stable_virtual_camera__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9854</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_terrestrial_landscape_0100">
      <source src="../../../assets/metric_examples/1c7dbcf5ff29__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.4535</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/356b8db1626b__image_static_stylized_passageways_0109.png" alt="Input image: image_static_stylized_passageways_0109" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.5003</div>
      <h4><code>image_static_stylized_passageways_0109</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit right. An indoor market passageway with shops, signs, and decorative lanterns.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_passageways_0109">
      <source src="../../../assets/metric_examples/738ebf610853__ac3d-image-static__image_static_stylized_passageways_0109.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9768</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_passageways_0109">
      <source src="../../../assets/metric_examples/a2ec606fa686__matrix_game3_5s__image_static_stylized_passageways_0109.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.4766</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/aaf3d40e89e2__image_static_stylized_living_spaces_0080.jpg" alt="Input image: image_static_stylized_living_spaces_0080" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 0.4898</div>
      <h4><code>image_static_stylized_living_spaces_0080</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pull out. A living room corner with bookshelf, potted plant, and yellow sectional sofa. Camera move left. A living room corner with framed photos, console table, and hallway. Camera pan left. A dining area with dining table, chairs, and wooden paneling.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_living_spaces_0080">
      <source src="../../../assets/metric_examples/7b9bb137b4d6__sana-wm-image-static__image_static_stylized_living_spaces_0080.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_living_spaces_0080">
      <source src="../../../assets/metric_examples/f87734cf7816__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_living_spaces_0080.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.5102</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d5e789a0106a__image_static_stylized_terrestrial_landscape_0157.jpg" alt="Input image: image_static_stylized_terrestrial_landscape_0157" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 0.4861</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0157</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit right. A stylized coastal landscape with lighthouse, cliff, and ocean.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_terrestrial_landscape_0157">
      <source src="../../../assets/metric_examples/5ad4100f15d0__ac3d-image-static__image_static_stylized_terrestrial_landscape_0157.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9780</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_terrestrial_landscape_0157">
      <source src="../../../assets/metric_examples/68293d09e9b9__sana-wm-image-static__image_static_stylized_terrestrial_landscape_0157.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.4919</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/84102b5b07bc__image_static_stylized_living_spaces_0139.png" alt="Input image: image_static_stylized_living_spaces_0139" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 0.4699</div>
      <h4><code>image_static_stylized_living_spaces_0139</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. A japanese living room with tatami mats, tea set, and low table.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_living_spaces_0139">
      <source src="../../../assets/metric_examples/2ef034707832__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_living_spaces_0139.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9885</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_living_spaces_0139">
      <source src="../../../assets/metric_examples/835e8a94f624__sana-wm-image-static__image_static_stylized_living_spaces_0139.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.5186</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/ba887a1cb180__image_static_stylized_dining_spaces_0152.png" alt="Input image: image_static_stylized_dining_spaces_0152" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 0.4648</div>
      <h4><code>image_static_stylized_dining_spaces_0152</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. A modern dining area with tables, decorative plants, and high chairs.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_dining_spaces_0152">
      <source src="../../../assets/metric_examples/eea20d0a3659__motionctrl__image_static_stylized_dining_spaces_0152.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_dining_spaces_0152">
      <source src="../../../assets/metric_examples/a8683614ef52__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_dining_spaces_0152.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.5352</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "LAION-Aesthetics Predictor (CLIP-based linear aesthetic regressor)"

    Used via pyiqa `laion_aes`; CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings.
