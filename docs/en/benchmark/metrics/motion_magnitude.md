# Motion Magnitude

Motion Magnitude quantifies **how much internal motion** a generated video contains, independent of motion correctness. It detects conservative generation—visually stable but nearly static outputs—without evaluating direction, semantics, or object behavior.

For adjacent frames \((I_t, I_{t+1})\), SEA-RAFT yields flow magnitude \(F_t(x,y) = \sqrt{u_t^2 + v_t^2}\). The frame-pair statistic is the pixel median; the raw score averages over pairs:

\[
S_{\mathrm{mag}}^{\mathrm{raw}} = \frac{1}{T-1}\sum_{t=0}^{T-2} \mathrm{median}_{x,y}\, F_t(x,y)
\tag{1}
\]

WorldArena z-score calibrates with \(\mu = 3.2425\), \(\sigma = 3.4505\), clip \([-0.8638, 2.7498]\), mapping to \([0.25, 0.75]\). Higher normalized scores indicate stronger visible motion. The metric is **official** when computation succeeds.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/3929c1e0d708__image_dynamic_photorealistic_articulated_0068.jpg" alt="Input image: image_dynamic_photorealistic_articulated_0068" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.7604</div>
      <h4><code>image_dynamic_photorealistic_articulated_0068</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic / articulated</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera fixed. The dog runs quickly across the grassy field, moving forward with alternating legs, open mouth, and perked ears while the fence and trees stay in place.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring dog changes pose and position substantially across the clip; the lower-scoring result repeats a much narrower pose and displacement range.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_dynamic_photorealistic_articulated_0068">
          <source src="../../../assets/metric_examples/curated_20260801/d473083aaa1b__image_dynamic_photorealistic_articulated_0068__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>motion_magnitude</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_dynamic_photorealistic_articulated_0068">
          <source src="../../../assets/metric_examples/aff317072a9d__matrix_game2_5s__image_dynamic_photorealistic_articulated_0068.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2396</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>motion_magnitude</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., ECCV 2024.
