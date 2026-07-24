# Long Sequence Imaging Quality

**long_sequence_imaging_quality** measures **mean no-reference technical image quality** (sharpness, noise, compression artifacts) over prediction frames using MUSIQ. Native scores are divided by $100$ and averaged to $[0,1]$.

\[
S_{\mathrm{img}} = \frac{1}{T}\sum_{t=1}^{T} s_t^{\mathrm{img}}
\tag{1}
\]

Higher is better. Long-rollout quality signal computed prediction-only via the `long_sequence` backend (`imaging_quality` dimension; default MUSIQ-SPAQ weights, longer side resized to $512$). See [Long-Sequence Metrics](long_sequence.md).

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `long_sequence_imaging_quality`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/13486b394bf9__image_static_photorealistic_verdant_landscape_0109.jpeg" alt="Input image: image_static_photorealistic_verdant_landscape_0109" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.4753</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0109</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. A forest edge with wildflowers and distant hills.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_verdant_landscape_0109">
      <source src="../../../assets/metric_examples/73755855521a__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_verdant_landscape_0109.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.8024</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_verdant_landscape_0109">
      <source src="../../../assets/metric_examples/4ac9274ee922__motionctrl__image_static_photorealistic_verdant_landscape_0109.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3270</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/8fcb15010df2__image_static_photorealistic_public_spaces_0068.png" alt="Input image: image_static_photorealistic_public_spaces_0068" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.4353</div>
      <h4><code>image_static_photorealistic_public_spaces_0068</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera push in. An abandoned chapel with altar, carvings, and faded symbols.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_public_spaces_0068">
      <source src="../../../assets/metric_examples/5009426ec26e__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.7406</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_public_spaces_0068">
      <source src="../../../assets/metric_examples/a14a53e19198__warp-as-history-image-static-8gpu-20260612_152000__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3053</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/4460d14379f4__image_static_stylized_suburb_0084.jpg" alt="Input image: image_static_stylized_suburb_0084" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 0.4182</div>
      <h4><code>image_static_stylized_suburb_0084</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pull out. A suburban intersection with trees, parked cars, and pedestrians.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_suburb_0084">
      <source src="../../../assets/metric_examples/0d348dbc85f1__motionctrl__image_static_stylized_suburb_0084.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.7578</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_suburb_0084">
      <source src="../../../assets/metric_examples/ac7c105f46d1__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_suburb_0084.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3397</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/72dc3ca43c46__image_static_stylized_verdant_landscape_0151.png" alt="Input image: image_static_stylized_verdant_landscape_0151" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 0.3910</div>
      <h4><code>image_static_stylized_verdant_landscape_0151</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move right. A rural landscape with path, tall grasses, and traditional buildings.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_verdant_landscape_0151">
      <source src="../../../assets/metric_examples/34239ebf5935__yume-sample5b-missing-reverse__image_static_stylized_verdant_landscape_0151.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.7832</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_verdant_landscape_0151">
      <source src="../../../assets/metric_examples/13ccb050b001__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_verdant_landscape_0151.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3922</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/266a02cd8d70__image_static_stylized_aquatic_landscape_0053.jpg" alt="Input image: image_static_stylized_aquatic_landscape_0053" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 0.3801</div>
      <h4><code>image_static_stylized_aquatic_landscape_0053</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move left. A rocky shoreline with vegetation, distant hills, and waves.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_aquatic_landscape_0053">
      <source src="../../../assets/metric_examples/cdd233ddb905__motionctrl__image_static_stylized_aquatic_landscape_0053.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.7774</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_aquatic_landscape_0053">
      <source src="../../../assets/metric_examples/51b05aeee0a9__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_aquatic_landscape_0053.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3974</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/03873b9fe77e__image_static_stylized_terrestrial_landscape_0149.png" alt="Input image: image_static_stylized_terrestrial_landscape_0149" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 0.3767</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0149</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan left. A desert landscape with dry shrubs, distant mesas, and rocky terrain. Camera move left. A desert landscape with distant mesas, rocky terrain, and dry shrubs. Camera pull out. A desert landscape with open sky, distant mesas, and rocky terrain.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_terrestrial_landscape_0149">
      <source src="../../../assets/metric_examples/476b02851a40__yume-sample5b-missing-reverse-5s-target__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.7517</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_terrestrial_landscape_0149">
      <source src="../../../assets/metric_examples/30b3f591dec6__matrix_game2_5s__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3750</div>
  </div>
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
