# Optical Flow AEPE

Optical Flow AEPE quantifies **short-horizon photometric consistency** in a generated video via forward–backward optical-flow round trips. It measures whether local appearance remains stable across consecutive frames—not whether motion matches ground truth.

For adjacent frames \(A\) and \(B\), sample points \(p_{A,i}\) in a center crop. Forward flow \(F_{A\to B}\) tracks each point to \(p_{B,i} = p_{A,i} + F_{A\to B}(p_{A,i})\); backward flow returns \(p'_{A,i} = p_{B,i} + F_{B\to A}(p_{B,i})\). The per-pair average end-point error is

\[
e_{\mathrm{AEPE}} = \frac{1}{N}\sum_{i=1}^{N} \left\| p_{A,i} - p'_{A,i} \right\|_2
\tag{1}
\]

The raw score averages \(e_{\mathrm{AEPE}}\) over all consecutive frame pairs. WorldArena reverse-normalizes with empirical range \([0, 1.192]\). Lower error (higher normalized score) indicates more stable local appearance. The metric is **official** when SEA-RAFT computation succeeds.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `optical_flow_aepe`
<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/fb408b9d36bc__image_static_stylized_terrestrial_landscape_0111.jpg" alt="Input image: image_static_stylized_terrestrial_landscape_0111" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.0000</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0111</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move left. A stylized mountain landscape with mountain slope, rocks, and vegetation.</p>
        <small>prompt source: prompt_target</small>
      </div>
  <p class="metric-case-note">No cross-model pair exists in the current formal snapshot; these are separate high/low extremes rather than a same-case model comparison.</p>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_terrestrial_landscape_0111">
      <source src="../../../assets/metric_examples/c396bdd2b691__ac3d-image-static__image_static_stylized_terrestrial_landscape_0111.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9841</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_verdant_landscape_0023">
      <source src="../../../assets/metric_examples/ae8594b1468b__ac3d-image-static__image_static_stylized_verdant_landscape_0023.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/3e63ea27eb16__image_static_photorealistic_terrestrial_landscape_0016.jpg" alt="Input image: image_static_photorealistic_terrestrial_landscape_0016" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.0000</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0016</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pull out. An expansive valley with forest, distant mountains, and sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
  <p class="metric-case-note">No cross-model pair exists in the current formal snapshot; these are separate high/low extremes rather than a same-case model comparison.</p>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_terrestrial_landscape_0016">
      <source src="../../../assets/metric_examples/438437dfd51c__ac3d-image-static__image_static_photorealistic_terrestrial_landscape_0016.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9830</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_living_spaces_0071">
      <source src="../../../assets/metric_examples/3bf118bc044f__ac3d-image-static__image_static_photorealistic_living_spaces_0071.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., ECCV 2024.
    [https://arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
