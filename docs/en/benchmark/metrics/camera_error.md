# Camera Error

Camera Error quantifies **trajectory fidelity** between a camera path recovered from a generated video and annotated ground-truth extrinsics. It measures viewpoint control compliance, not object motion or instruction following.

Let \(\{\mathbf{T}^{\mathrm{gt}}_t\}_{t=1}^{N}\) and \(\{\mathbf{T}^{\mathrm{pred}}_t\}_{t=1}^{N}\) denote aligned $4 \times 4$ camera-to-world matrices from annotations and from visual odometry (MegaSAM or DROID-SLAM). Decompose rotation \(\mathbf{R}_t\) and translation \(\mathbf{t}_t\) from each matrix. Translation is scale-aligned by solving

\[
c^\star = \arg\min_{c} \sum_{t=1}^{N} \left\|\mathbf{t}^{\mathrm{gt}}_t - c\,\mathbf{t}^{\mathrm{pred}}_t\right\|_2
\tag{1}
\]

with a mean-distance ratio fallback when the convex program is unavailable. Mean translation error is

\[
E_t = \frac{1}{N}\sum_{t=1}^{N} \left\|c^\star \mathbf{t}^{\mathrm{pred}}_t - \mathbf{t}^{\mathrm{gt}}_t\right\|_2
\tag{2}
\]

Mean rotation error (degrees) uses relative rotation \(\mathbf{R}^{\mathrm{rel}}_t = \mathbf{R}^{\mathrm{pred}}_t (\mathbf{R}^{\mathrm{gt}}_t)^\top\):

\[
E_r = \frac{1}{N}\sum_{t=1}^{N} \arccos\!\left(\mathrm{clip}\!\left(\frac{\mathrm{tr}(\mathbf{R}^{\mathrm{rel}}_t) - 1}{2}, -1, 1\right)\right) \cdot \frac{180}{\pi}
\tag{3}
\]

WorldArena normalizes each component to \([0,1]\) with empirical ranges \([0, 15°]\) for rotation and \([0, 0.5]\) for translation (lower is better), then combines them by the **geometric mean** to yield \(S_{\mathrm{cam}} \in [0,1]\). Higher \(S_{\mathrm{cam}}\) indicates closer camera control. The metric is **official** when pose annotations or protocol-aligned synthetic camera paths are available.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `camera_error`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d6b1f731ad67__image_static_photorealistic_aquatic_landscape_0001.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0001" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.9189</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit left. A coastal landscape with ocean waves, distant hills, and sunset sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_aquatic_landscape_0001">
      <source src="../../../assets/metric_examples/1a70929b6c9a__camera_error_megasam_smoke_dreamx__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9189</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_aquatic_landscape_0001">
      <source src="../../../assets/metric_examples/56c1b531c0e3__camera_error_corrected_smoke_dreamx__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "DROID-SLAM — Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Teed & Deng, NeurIPS 2021.
    [https://arxiv.org/abs/2108.10869](https://arxiv.org/abs/2108.10869)
