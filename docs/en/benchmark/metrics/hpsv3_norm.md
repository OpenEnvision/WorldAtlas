# HPSv3-Norm

HPSv3-Norm measures **human-preference-aligned visual quality** of a generated video using the HPSv3 reward model. It is **prediction-only**: each evaluated frame is scored independently, without pairing against a reference rollout.

`hpsv3_norm` is an official `image_static` quality metric and enters both `per_metric_summary` and the equal-weight seven-metric quality total.

WorldArena uniformly subsamples up to 20 frames from the prediction video. For frame $x_i$, the HPSv3 inferencer outputs a raw reward $r_i$. The per-video mean raw reward is

\[
\bar{r}
=
\frac{1}{N}\sum_{i=1}^{N} r_i
\tag{1}
\]

Let $p_1$ and $p_{99}$ denote fixed normalization percentiles (defaults: $p_1 = 5.21$, $p_{99} = 8.66$). The normalized score on $[0, 100]$ is

\[
S_{\mathrm{HPS}}
=
\mathrm{clip}\!\left(
\frac{\bar{r} - p_1}{p_{99} - p_1} \times 100,\,
0,\,
100
\right)
\tag{2}
\]

WorldArena stores the metric raw value in $[0, 1]$ as $S_{\mathrm{HPS}} / 100$. Higher values indicate stronger human-preference alignment. When the HPSv3 backend or checkpoints are unavailable, the metric is reported as `not_applicable`.

## References {#reference}

!!! quote "HPSv3 — Human Preference Score"

    Vision-language reward model used to score image quality from human-preference supervision; WorldArena applies linear percentile normalization to per-video mean raw rewards.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d6b1f731ad67__image_static_photorealistic_aquatic_landscape_0001.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0001" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.1702</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera orbit left. A coastal landscape with ocean waves, distant hills, and sunset sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout remains a coherent, detailed sunset landscape; the lower-scoring rollout is flatter and less preference-aligned. HPSv3 scores prediction frames only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_aquatic_landscape_0001">
          <source src="../../../assets/metric_examples/e3f37ff52cf2__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.1702</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_aquatic_landscape_0001">
          <source src="../../../assets/metric_examples/curated_20260801/9d372e4f2066__image_static_photorealistic_aquatic_landscape_0001__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
