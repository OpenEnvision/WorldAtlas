# Long-Sequence Metrics

WorldArena bundles a native `Long-Sequence` backend to extract long-horizon motion and quality signals from full prediction videos. These metrics do not only inspect single frames or adjacent frame pairs. They split a generated video into short clips, then analyze longer-range motion continuity and visual quality.

The main question is whether a generated video still preserves motion continuity and image quality over a longer horizon.

These metrics are computed prediction-only. They do not use the benchmark reference rollout or annotations. They are best used to explain long-video degradation modes such as motion collapse, abrupt temporal artifacts, or quality decay over time.

## Covered Metrics

| Metric name | Wrapped Long-Sequence dimension | Role in WorldArena |
| --- | --- | --- |
| `long_sequence_motion_smoothness` | `motion_smoothness` | motion smoothness based on video-frame-interpolation error |
| `long_sequence_dynamic_degree` | `dynamic_degree` | optical-flow-based probe for whether enough motion occurs |
| `long_sequence_aesthetic_quality` | `aesthetic_quality` | no-reference aesthetic quality |
| `long_sequence_imaging_quality` | `imaging_quality` | MUSIQ-based image clarity and degradation quality |

## Principle

Short-range metrics can miss failures that appear only after a longer rollout. The Long-Sequence backend first splits the full video into short clips, then applies native long-video routines to motion and quality dimensions.

`motion_smoothness` uses AMT-S to interpolate middle frames and compares the interpolation with real middle frames. `dynamic_degree` uses RAFT flow strength to detect whether motion is sufficiently present. `aesthetic_quality` and `imaging_quality` use no-reference quality models over clip frames.

## Detailed Algorithm

1. WorldArena reads the full prediction video and prepares a Long-Sequence input workspace.
2. `worldarena.benchmark.long_video` splits the video into `split_clip/` inputs and builds the minimal manifest required by each dimension.
3. A dispatcher routes each requested dimension to the corresponding implementation:
    - `motion_smoothness`: AMT-S interpolation, comparing predicted middle frames with real middle frames.
    - `dynamic_degree`: RAFT optical flow, counting whether high-motion transitions exceed the threshold.
    - `aesthetic_quality`: no-reference aesthetic scorer over clip frames.
    - `imaging_quality`: MUSIQ over clip frames.
4. Each WorldArena metric reads only its own wrapped dimension.
5. The wrapper records runtime metadata, including device, source root, clip count, per-video scores, and dimension-specific details.
6. Each metric is independently normalized according to the `[0, 1]` range in `config/benchmark.yaml`; all four metrics are higher-is-better.

## Key Formulas

For `motion_smoothness`, AMT-S interpolates middle frames from evenly sampled frames and compares interpolated and real middle frames using mean absolute difference:

!!! info "Motion smoothness"

    \[
    s_{\mathrm{smooth}}
    =
    \operatorname{clamp}
    \left(
        \frac{255-\operatorname{mean}(|I^{real}_{mid}-I^{interp}_{mid}|)}{255},
        0,
        1
    \right)
    \]

For `dynamic_degree`, RAFT estimates optical flow between adjacent sampled frames and uses the mean magnitude of the top 5% highest-motion pixels as the transition motion score. If enough transitions exceed the motion threshold, the clip is marked dynamic and outputs `1.0`; otherwise it outputs `0.0`.

For `aesthetic_quality` and `imaging_quality`, the backend scores frames in each clip with no-reference quality models, then averages over clips, videos, and the sample batch.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `long_sequence_aesthetic_quality`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/b3d0e7407370__image_static_stylized_terrestrial_landscape_0004.jpg" alt="Input image: image_static_stylized_terrestrial_landscape_0004" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.3237</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0004</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move right. A snowy forest with snow-covered trees, distant path, and snowy ground.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_terrestrial_landscape_0004">
      <source src="../../../assets/metric_examples/95a9cc54e18b__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_terrestrial_landscape_0004.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.6700</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_terrestrial_landscape_0004">
      <source src="../../../assets/metric_examples/cf80f8e9ac0c__motionctrl__image_static_stylized_terrestrial_landscape_0004.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3464</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f26edd43f07f__image_static_stylized_passageways_0105.png" alt="Input image: image_static_stylized_passageways_0105" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.2892</div>
      <h4><code>image_static_stylized_passageways_0105</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera push in. A stylized indoor passageway with wooden floor, bookshelves, and reading area.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_passageways_0105">
      <source src="../../../assets/metric_examples/f8f86fde3531__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_passageways_0105.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.5670</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_passageways_0105">
      <source src="../../../assets/metric_examples/58ddbf8228dd__motionctrl__image_static_stylized_passageways_0105.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2777</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2c2b9c480799__image_static_photorealistic_terrestrial_landscape_0100.jpg" alt="Input image: image_static_photorealistic_terrestrial_landscape_0100" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 0.2665</div>
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
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.6438</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_terrestrial_landscape_0100">
      <source src="../../../assets/metric_examples/b61a742a411b__motionctrl__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3773</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/1a7dd6bacd65__image_static_photorealistic_suburb_0063.png" alt="Input image: image_static_photorealistic_suburb_0063" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 0.2628</div>
      <h4><code>image_static_photorealistic_suburb_0063</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan right. A suburban street with houses, parked cars, and street signs.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_suburb_0063">
      <source src="../../../assets/metric_examples/2ad22bc01a86__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_suburb_0063.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.5978</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_suburb_0063">
      <source src="../../../assets/metric_examples/ebf0aaa3ac2b__motionctrl__image_static_photorealistic_suburb_0063.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3350</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/810078323779__image_static_photorealistic_verdant_landscape_0080.jpg" alt="Input image: image_static_photorealistic_verdant_landscape_0080" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 0.2586</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0080</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit left. An overgrown greenhouse with vines, rusted tools, and workbenches.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_verdant_landscape_0080">
      <source src="../../../assets/metric_examples/1f27af206163__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_verdant_landscape_0080.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.6437</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_verdant_landscape_0080">
      <source src="../../../assets/metric_examples/b8d768559d11__motionctrl__image_static_photorealistic_verdant_landscape_0080.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.3851</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/a68afb03ba7f__image_static_stylized_public_spaces_0115.png" alt="Input image: image_static_stylized_public_spaces_0115" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 0.2425</div>
      <h4><code>image_static_stylized_public_spaces_0115</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan left. A vintage museum with display cases, historical artifacts, and inventions.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_public_spaces_0115">
      <source src="../../../assets/metric_examples/963841dad8d8__stable_virtual_camera__image_static_stylized_public_spaces_0115.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.6808</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_public_spaces_0115">
      <source src="../../../assets/metric_examples/dfa37f7aa924__motionctrl__image_static_stylized_public_spaces_0115.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.4384</div>
  </div>
</figure>
  </div>
</article>
</div>
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
