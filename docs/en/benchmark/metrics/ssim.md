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
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/205a86ec0eb6__image_static_photorealistic_living_spaces_0069__input.jpg" alt="Input image: image_static_photorealistic_living_spaces_0069" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.8650</div>
      <h4><code>image_static_photorealistic_living_spaces_0069</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pull out. A living room corner with bookshelf, floor lamp, and small side table. Camera pan right. A living room corner with framed photos, coat rack, and small rug. Camera move right. A dining area with dining table, chairs, and hanging light fixture.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout retains the input room's spatial layout and major structures; the lower-scoring rollout drifts to generic, differently arranged interiors. SSIM emphasizes structural fidelity to the reference.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_living_spaces_0069">
          <source src="../../../assets/metric_examples/curated_20260801/b2598580eefe__image_static_photorealistic_living_spaces_0069__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9625</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_living_spaces_0069">
          <source src="../../../assets/metric_examples/curated_20260801/a50c49b082e2__image_static_photorealistic_living_spaces_0069__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0975</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
