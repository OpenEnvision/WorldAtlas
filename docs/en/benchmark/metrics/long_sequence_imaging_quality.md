# Long Sequence Imaging Quality

**long_sequence_imaging_quality** measures **mean no-reference technical image quality** (sharpness, noise, compression artifacts) over prediction frames using MUSIQ. Native scores are divided by $100$ and averaged to $[0,1]$.

\[
S_{\mathrm{img}} = \frac{1}{T}\sum_{t=1}^{T} s_t^{\mathrm{img}}
\tag{1}
\]

Higher is better. Long-rollout quality signal computed prediction-only via the `long_sequence` backend (`imaging_quality` dimension; default MUSIQ-SPAQ weights, longer side resized to $512$). See [Long-Sequence Metrics](long_sequence.md).

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/297e8735b260__image_static_photorealistic_public_spaces_0068__input.jpg" alt="Input image: image_static_photorealistic_public_spaces_0068" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.4353</div>
      <h4><code>image_static_photorealistic_public_spaces_0068</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera push in. An abandoned chapel with altar, carvings, and faded symbols.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_public_spaces_0068">
          <source src="../../../assets/metric_examples/5009426ec26e__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7406</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_public_spaces_0068">
          <source src="../../../assets/metric_examples/a14a53e19198__warp-as-history-image-static-8gpu-20260612_152000__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3053</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/8ff65b592716__image_static_stylized_terrestrial_landscape_0149__input.jpg" alt="Input image: image_static_stylized_terrestrial_landscape_0149" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.3767</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0149</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan left. A desert landscape with dry shrubs, distant mesas, and rocky terrain. Camera move left. A desert landscape with distant mesas, rocky terrain, and dry shrubs. Camera pull out. A desert landscape with open sky, distant mesas, and rocky terrain.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_terrestrial_landscape_0149">
          <source src="../../../assets/metric_examples/476b02851a40__yume-sample5b-missing-reverse-5s-target__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7517</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_terrestrial_landscape_0149">
          <source src="../../../assets/metric_examples/30b3f591dec6__matrix_game2_5s__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3750</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "MUSIQ: Multi-scale Image Quality Transformer"

    Zejiang Hou, Hui Ren, Ming Sun, Jiahao Li, Weining Wang.  
    ICCV 2021.  
    [https://arxiv.org/abs/2108.05997](https://arxiv.org/abs/2108.05997)
