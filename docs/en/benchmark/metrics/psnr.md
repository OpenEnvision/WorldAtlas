# PSNR

**psnr** measures frame-wise signal fidelity between prediction and reference.

## What It Measures

- pixel-level similarity to the reference
- reconstruction fidelity, not semantic correctness
- whether the generated frames stay numerically close to the reference frames

## Detailed Algorithm

1. WorldArena pairs sampled prediction and reference frames at aligned time indices.
2. The prediction frame is resized to the reference frame resolution if needed.
3. For each frame pair, WorldArena computes
  `PSNR = 20 * log10(255) - 10 * log10(MSE)`.
4. Perfect matches are clamped to `100.0` instead of infinity.
5. The raw metric is the mean frame PSNR.

## Inputs And Outputs


| Item              | Current implementation                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Prediction input  | sampled prediction frames                                                                         |
| Reference input   | sampled reference frames                                                                          |
| Annotation input  | none                                                                                              |
| Raw output        | one scalar mean PSNR value                                                                        |
| Normalized output | min/max scaled score from `config/benchmark.yaml`                                      |
| Details payload   | `frame_scores`, frame index metadata                                                              |
| Failure mode      | returns `raw = null`, `normalized = null`, and an error string if no aligned frames are available |


## Integration Readiness


| Item                | Verdict           |
| ------------------- | ----------------- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |


- Why it is useful: it is simple, stable, and easy to interpret.
- Why it is limited: it strongly rewards copying the reference and says little about semantic validity.
- Recommended use: keep it as a reconstruction-style appendix metric only.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d943a1dd2b07__image_static_photorealistic_workspaces_0024__input.jpg" alt="Input image: image_static_photorealistic_workspaces_0024" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.9382</div>
      <h4><code>image_static_photorealistic_workspaces_0024</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pull out. A minimalist workspace with desk, chair, and lamp. Camera pan right. A minimalist workspace with hallway, door, and wall art. Camera move right. A minimalist workspace with hallway, wall art, and small table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout preserves the input workspace's structure and appearance across the move; the lower-scoring rollout replaces it with unrelated close-ups. PSNR measures reference fidelity, not standalone aesthetics.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_workspaces_0024">
          <source src="../../../assets/metric_examples/curated_20260801/5339bb13b9a3__image_static_photorealistic_workspaces_0024__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9449</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>psnr</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_workspaces_0024">
          <source src="../../../assets/metric_examples/curated_20260801/4b869c9aa35b__image_static_photorealistic_workspaces_0024__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0067</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>psnr</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
