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
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/31dd4f9d844e__image_dynamic_photorealistic_articulated_0061__input.jpg" alt="Input image: image_dynamic_photorealistic_articulated_0061" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8369</div>
      <h4><code>image_dynamic_photorealistic_articulated_0061</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic / articulated</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera fixed. Two tigers stir on the rocky ledges; the upper tiger stretches a foreleg while the lower tiger turns its head and flicks its ears. The rugged enclosure rocks stay unchanged.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_dynamic_photorealistic_articulated_0061">
          <source src="../../../assets/metric_examples/curated_20260801/d37481d48321__image_dynamic_photorealistic_articulated_0061__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.8448</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>motion_accuracy</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_dynamic_photorealistic_articulated_0061">
          <source src="../../../assets/metric_examples/curated_20260801/5763f5cf2870__image_dynamic_photorealistic_articulated_0061__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0079</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>motion_accuracy</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/1d0cfde448b9__image_dynamic_stylized_deformable_0031__input.jpg" alt="Input image: image_dynamic_stylized_deformable_0031" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.7946</div>
      <h4><code>image_dynamic_stylized_deformable_0031</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized / deformable</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera fixed. A blocky octopus emerges from the pixelated water near the sandy shore, its tentacles spread outward around its large eyes. The water, grass, and blocky trees remain calm.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_dynamic_stylized_deformable_0031">
          <source src="../../../assets/metric_examples/curated_20260801/fb6e4230eb82__image_dynamic_stylized_deformable_0031__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9839</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>motion_accuracy</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_dynamic_stylized_deformable_0031">
          <source src="../../../assets/metric_examples/curated_20260801/8c91865b7966__image_dynamic_stylized_deformable_0031__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.1893</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>motion_accuracy</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/aa24ac0a445a__image_dynamic_stylized_articulated_0020__input.jpg" alt="Input image: image_dynamic_stylized_articulated_0020" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8605</div>
      <h4><code>image_dynamic_stylized_articulated_0020</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized / articulated</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera fixed. Two stylized sharks glide over the underwater landscape, their tails sweeping gently as they pass seaweed and rocky formations. Sunlight filters through the clear water while the seabed remains steady.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
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
