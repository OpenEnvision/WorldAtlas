# Rollout Aesthetic Drift

**rollout_aesthetic_drift** measures **temporal aesthetic degradation** along a rollout. Per-frame LAION aesthetic scores are split into $N=10$ equal segments; drift is the relative drop from the best to the worst segment mean.

\[
D_{\mathrm{aes}} = \frac{\bar{s}_{\mathrm{best}} - \bar{s}_{\mathrm{worst}}}{|\bar{s}_{\mathrm{best}}|}
\tag{1}
\]

Lower is better. The value is clamped to $[0,1]$. Computed prediction-only. Backend: `quality_drift` (aesthetic component via pyiqa `laion_aes`).

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/08a7ae54eb87__image_static_stylized_living_spaces_0080__input.jpg" alt="Input image: image_static_stylized_living_spaces_0080" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.4898</div>
      <h4><code>image_static_stylized_living_spaces_0080</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pull out. A living room corner with bookshelf, potted plant, and yellow sectional sofa. Camera move left. A living room corner with framed photos, console table, and hallway. Camera pan left. A dining area with dining table, chairs, and wooden paneling.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0080">
          <source src="../../../assets/metric_examples/7b9bb137b4d6__sana-wm-image-static__image_static_stylized_living_spaces_0080.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0080">
          <source src="../../../assets/metric_examples/f87734cf7816__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_living_spaces_0080.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.5102</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/27ef0171b751__image_static_photorealistic_terrestrial_landscape_0100__input.jpg" alt="Input image: image_static_photorealistic_terrestrial_landscape_0100" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.5320</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0100</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A mountain landscape with valley, river, and mountain peaks. Camera pull out. A mountain landscape with forest, wildlife, and deer. Camera move right. A mountain landscape with village, smoke, and chimneys.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_terrestrial_landscape_0100">
          <source src="../../../assets/metric_examples/3ef2096fbf6e__stable_virtual_camera__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9854</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_terrestrial_landscape_0100">
          <source src="../../../assets/metric_examples/1c7dbcf5ff29__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.4535</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/2cd9a570f49f__image_static_stylized_living_spaces_0139__input.jpg" alt="Input image: image_static_stylized_living_spaces_0139" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.4699</div>
      <h4><code>image_static_stylized_living_spaces_0139</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A japanese living room with tatami mats, tea set, and low table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0139">
          <source src="../../../assets/metric_examples/2ef034707832__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_living_spaces_0139.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9885</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0139">
          <source src="../../../assets/metric_examples/835e8a94f624__sana-wm-image-static__image_static_stylized_living_spaces_0139.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.5186</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "LAION-Aesthetics Predictor (CLIP-based linear aesthetic regressor)"

    Used via pyiqa `laion_aes`; CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings.
