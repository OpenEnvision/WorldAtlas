# SSIM

**ssim** measures frame-wise structural similarity between prediction and reference.

## What It Measures

- structural closeness to the reference
- luminance, contrast, and local-structure agreement
- reconstruction-style similarity with a mild perceptual bias

## Detailed Algorithm

1. WorldArena pairs sampled prediction and reference frames at aligned time indices.
2. The prediction frame is resized to the reference frame resolution if needed.
3. For each RGB channel, it computes a standard Gaussian-window SSIM map.
4. Channel SSIM scores are averaged into one frame score.
5. The raw metric is the mean frame SSIM.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | sampled reference frames |
| Annotation input | none |
| Raw output | one scalar mean SSIM score in `[0, 1]` |
| Normalized output | bounded scalar under the default config |
| Details payload | `frame_scores`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if no aligned frames are available |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it is more structure-aware than raw PSNR.
- Why it is limited: like PSNR, it still rewards reference copying and should not stand in for semantic
  or control correctness.
- Recommended use: report it only as a supporting full-reference metric.

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c2cb9cbb7479__image_static_photorealistic_passageways_0076__input.jpg" alt="Input image: image_static_photorealistic_passageways_0076" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.7476</div>
      <h4><code>image_static_photorealistic_passageways_0076</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. A library aisle with bookshelves, books, and reading table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0076">
          <source src="../../../assets/metric_examples/curated_20260801/5bc5be0ae204__image_static_photorealistic_passageways_0076__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9410</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0076">
          <source src="../../../assets/metric_examples/curated_20260801/07118e8eb56d__image_static_photorealistic_passageways_0076__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.1934</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/5bb59c6bfb78__image_static_photorealistic_dining_spaces_0032__input.jpg" alt="Input image: image_static_photorealistic_dining_spaces_0032" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.6786</div>
      <h4><code>image_static_photorealistic_dining_spaces_0032</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. An eclectic kitchen with martini sign, posters, and fruit bowl.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_dining_spaces_0032">
          <source src="../../../assets/metric_examples/curated_20260801/c0df8ae0b4fc__image_static_photorealistic_dining_spaces_0032__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9218</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_dining_spaces_0032">
          <source src="../../../assets/metric_examples/curated_20260801/81c3f89d947c__image_static_photorealistic_dining_spaces_0032__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2432</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/e31f9eb4b75b__image_static_photorealistic_verdant_landscape_0001__input.jpg" alt="Input image: image_static_photorealistic_verdant_landscape_0001" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.6761</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pull out. An indoor garden with flower beds, statues, and arched windows. Camera pan left. An indoor garden with side path, benches, and flower beds. Camera move left. An indoor garden with exit door, plants, and sunlight.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_verdant_landscape_0001">
          <source src="../../../assets/metric_examples/curated_20260801/3b4d97b32e5b__image_static_photorealistic_verdant_landscape_0001__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7518</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_verdant_landscape_0001">
          <source src="../../../assets/metric_examples/curated_20260801/258b319b48d6__image_static_photorealistic_verdant_landscape_0001__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0757</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
