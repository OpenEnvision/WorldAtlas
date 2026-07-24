# Depth Collision

Depth Collision is a binary flag for implausibly near geometry directly ahead of the camera. Video Depth Anything supplies per-frame metric depth; a center crop tests whether the forward viewing corridor becomes abnormally close during rollout.

For frame \(t\) with depth map \(D_t\) and center crop \(\mathcal{C}\),

\[
\bar{d}_t = \frac{1}{|\mathcal{C}|}\sum_{(x,y)\in\mathcal{C}} D_t(x,y)
\tag{1}
\]

Given threshold \(\tau\) (default \(1.0\)), the collision indicator is

\[
e_{\mathrm{collision}} = \mathbf{1}\!\left[\min_t \bar{d}_t < \tau\right]
\tag{2}
\]

Raw score \(e_{\mathrm{collision}} \in \{0,1\}\) is reverse-normalized so that no collision maps to higher score. The flag proxies near-depth failure (wall penetration, face-close motion) under estimated depth—it is not a physics-engine contact test.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `depth_collision`
<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d6b1f731ad67__image_static_photorealistic_aquatic_landscape_0001.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0001" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · extreme case from current snapshot</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit left. A coastal landscape with ocean waves, distant hills, and sunset sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
  <p class="metric-case-note">No cross-model pair with the same input image and same prompt is available for this metric in the current formal snapshot; this exhibit is a single high/low extreme case.</p>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_aquatic_landscape_0001">
      <source src="../../../assets/metric_examples/e3f37ff52cf2__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_aquatic_landscape_0001">
      <source src="../../../assets/metric_examples/e3f37ff52cf2__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 1.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/20f6a24367ac__image_static_photorealistic_aquatic_landscape_0017.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0017" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · extreme case from current snapshot</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0017</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit left. An aquatic landscape with red lighthouse, ship&#x27;s deck, and harbor.</p>
        <small>prompt source: prompt_target</small>
      </div>
  <p class="metric-case-note">No cross-model pair with the same input image and same prompt is available for this metric in the current formal snapshot; this exhibit is a single high/low extreme case.</p>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_aquatic_landscape_0017">
      <source src="../../../assets/metric_examples/d339e4209040__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0017.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_aquatic_landscape_0017">
      <source src="../../../assets/metric_examples/d339e4209040__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0017.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 1.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/4819b8b280f4__image_static_photorealistic_aquatic_landscape_0033.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0033" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · extreme case from current snapshot</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0033</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move right. A coastal landscape with waves, beach, and vegetation.</p>
        <small>prompt source: prompt_target</small>
      </div>
  <p class="metric-case-note">No cross-model pair with the same input image and same prompt is available for this metric in the current formal snapshot; this exhibit is a single high/low extreme case.</p>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_aquatic_landscape_0033">
      <source src="../../../assets/metric_examples/f9f24c8499ad__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0033.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_aquatic_landscape_0033">
      <source src="../../../assets/metric_examples/f9f24c8499ad__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0033.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 1.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"

    Chen et al., arXiv:2501.12375, 2025.
