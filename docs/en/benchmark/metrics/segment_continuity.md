# Segment Continuity

Segment Continuity measures whether a generated video contains **detected scene cuts**. Unintended hard transitions break temporal coherence and indicate segment discontinuities.

TransNetV2 outputs a per-frame scene-boundary probability $p_i$ for decoded frame $x_i$. Frames with $p_i > \theta$ are flagged as cut candidates, with defaults $\theta = 0.5$. A cut is recorded at frame index $i$ when the binary mask rises from $0$ to $1$ at $i$, following TransNetV2 transition parsing.

To suppress spurious detections, cut candidates are filtered so that every retained scene spans at least $L$ frames (default $L = 10$). Let $n_{\mathrm{cut}}$ denote the number of retained cuts in one video. The per-video score is binary:

\[
S_{\mathrm{seg}}
=
\begin{cases}
1, & n_{\mathrm{cut}} = 0 \\
0, & n_{\mathrm{cut}} > 0
\end{cases}
\tag{1}
\]

Across $N$ evaluated videos, the model-level aggregate is

\[
\bar{S}_{\mathrm{seg}}
=
1 - \frac{n_{\mathrm{cut}}}{N}
\tag{2}
\]

where $n_{\mathrm{cut}}$ counts videos with at least one retained cut. WorldArena stores the per-sample binary value from Eq. (1) as the metric raw score. The metric is **prediction-only** and requires TransNetV2 inference code and weights.

## References {#reference}

!!! quote "TransNet V2 — Shot Boundary Detection"

    Sočik et al., ACM MM 2020.
    [https://arxiv.org/abs/2008.04838](https://arxiv.org/abs/2008.04838)

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c5ed17f8111b__image_static_photorealistic_aquatic_landscape_0044__input.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0044" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 1.0000</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0044</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pan left. An aquatic landscape with shoreline, seashells, and distant boat. Camera pull out. An aquatic landscape with sand dunes, beach grass, and seagulls. Camera move left. An aquatic landscape with boardwalk, beachgoers, and umbrellas.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring beach rollout stays within one continuous scene; the lower-scoring rollout contains a detected hard cut from the beach hut to an unrelated harbor scene.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_aquatic_landscape_0044">
          <source src="../../../assets/metric_examples/curated_20260801/b0f27193421a__image_static_photorealistic_aquatic_landscape_0044__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>segment_continuity</span><span>Detected hard cuts: 0</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_aquatic_landscape_0044">
          <source src="../../../assets/metric_examples/curated_20260801/665ac6c76204__image_static_photorealistic_aquatic_landscape_0044__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>segment_continuity</span><span>Detected hard cuts: 1</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
