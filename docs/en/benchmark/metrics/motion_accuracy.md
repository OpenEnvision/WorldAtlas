# Motion Accuracy

Motion Accuracy measures **foreground motion salience** inside supervised dynamic regions. SAM2 propagates initial segmentation masks through the prediction; SEA-RAFT estimates optical flow between consecutive frames. The metric tests whether masked object regions exhibit stronger motion than the background—it does **not** compare realized motion to a target trajectory, direction, or velocity profile.

For consecutive frame pair \((I_t, I_{t+1})\), let \(\mathbf{u}_t\) be the optical-flow field and \(M_t\) the propagated (dilated) object mask. Define magnitude \(m_t = \|\mathbf{u}_t\|_2\) and

\[
\Delta_t = \max_{p \in M_t} m_t(p) - \max_{p \notin M_t} m_t(p)
\tag{1}
\]

The raw score is the mean salience over all pairs:

\[
S_{\mathrm{mot}}^{\mathrm{raw}} = \frac{1}{N-1}\sum_{t=1}^{N-1} \Delta_t
\tag{2}
\]

WorldArena applies z-score calibration with \(\mu = -0.1965\), \(\sigma = 2.3687\), clip bounds \([-1.6147, 1.5180]\), and maps to \([0.25, 0.75]\). Higher normalized scores indicate stronger mask-conditioned motion relative to background. The metric is **official** when per-frame segmentation masks are available under `sample.mask_path`.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/ae595b686cab__image_dynamic_stylized_articulated_0020__input.jpg" alt="Input image: image_dynamic_stylized_articulated_0020" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.8605</div>
      <h4><code>image_dynamic_stylized_articulated_0020</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized / articulated</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera fixed. Two stylized sharks glide over the underwater landscape, their tails sweeping gently as they pass seaweed and rocky formations. Sunlight filters through the clear water while the seabed remains steady.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout keeps both sharks as salient moving foreground objects against a steadier seabed; the lower-scoring rollout mixes object deformation with broader scene motion.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_dynamic_stylized_articulated_0020">
          <source src="../../../assets/metric_examples/curated_20260801/ceb76b5f7507__image_dynamic_stylized_articulated_0020__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9535</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>motion_accuracy</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_dynamic_stylized_articulated_0020">
          <source src="../../../assets/metric_examples/curated_20260801/8f5a70ee480e__image_dynamic_stylized_articulated_0020__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0930</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>motion_accuracy</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "SAM 2 — Segment Anything in Images and Videos"

    Ravi et al., 2024.
    [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

!!! quote "SEA-RAFT — Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., 2024.
    [https://arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
