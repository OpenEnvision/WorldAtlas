# Background Consistency

Background Consistency measures **temporal stability of scene appearance** in a generated video. Large shifts in background layout, texture, or global scene context between adjacent frames lower the score.

For each decoded frame $x_i$, CLIP ViT-B/32 extracts a normalized embedding $f_i$. The metric averages cosine similarity over consecutive frame pairs:

\[
S_{\mathrm{bg}}
=
\frac{1}{N-1}\sum_{i=1}^{N-1}\cos(f_i, f_{i+1})
\tag{1}
\]

where $N$ is the number of evaluated frames and $\cos(\cdot,\cdot)$ is cosine similarity between L2-normalized CLIP image features. WorldArena clamps $S_{\mathrm{bg}}$ to $[0, 1]$; higher values indicate more stable backgrounds. The metric is **prediction-only** and requires at least two frames.

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/340a8644362b__image_static_photorealistic_city_0043__input.jpg" alt="Input image: image_static_photorealistic_city_0043" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.1517</div>
      <h4><code>image_static_photorealistic_city_0043</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. A narrow city street with historic buildings, street corner, and streetlights.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_city_0043">
          <source src="../../../assets/metric_examples/curated_20260801/48e0a8e64f87__image_static_photorealistic_city_0043__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9849</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.9849</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_city_0043">
          <source src="../../../assets/metric_examples/curated_20260801/cafc8cbd4022__image_static_photorealistic_city_0043__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.8333</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.8333</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/6889d948d9ba__image_static_photorealistic_city_0041__input.jpg" alt="Input image: image_static_photorealistic_city_0041" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.1444</div>
      <h4><code>image_static_photorealistic_city_0041</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. An old industrial building with adjacent street, buildings, and parked cars.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_city_0041">
          <source src="../../../assets/metric_examples/curated_20260801/346b0f1c5c3b__image_static_photorealistic_city_0041__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9882</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.9882</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_city_0041">
          <source src="../../../assets/metric_examples/curated_20260801/49d5f2e01c18__image_static_photorealistic_city_0041__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.8437</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.8437</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/087c40c40545__image_static_stylized_aquatic_landscape_0063__input.jpg" alt="Input image: image_static_stylized_aquatic_landscape_0063" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.1793</div>
      <h4><code>image_static_stylized_aquatic_landscape_0063</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. A stylized lake with trees, hills, and clouds.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_aquatic_landscape_0063">
          <source src="../../../assets/metric_examples/curated_20260801/a285b51b62d0__image_static_stylized_aquatic_landscape_0063__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9933</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.9933</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_aquatic_landscape_0063">
          <source src="../../../assets/metric_examples/curated_20260801/85e93576194e__image_static_stylized_aquatic_landscape_0063__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.8140</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.8140</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
