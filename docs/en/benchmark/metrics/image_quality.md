# Image Quality (MUSIQ)

`image_quality` is an official no-reference `image_static` quality metric. It evaluates prediction frames with the MUSIQ-SPAQ image-quality model and does not require a paired reference rollout.

For each protocol-selected prediction frame $I_i$, MUSIQ produces a native score $q_i$ on its $[0,100]$ scale. WorldArena normalizes and clips each score,

\[
s_i = \mathrm{clip}\!\left(\frac{q_i}{100}, 0, 1\right),
\]

then averages the normalized frame scores:

\[
S_{\mathrm{MUSIQ}} = \frac{1}{N}\sum_{i=1}^{N}s_i.
\]

Higher is better. The resulting value enters `per_metric_summary` and the equal-weight seven-metric quality total. The default backend is `musiq`, with `longer` preprocessing.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/4bfa2a1518b3__image_static_stylized_suburb_0082__input.jpg" alt="Input image: image_static_stylized_suburb_0082" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.5350</div>
      <h4><code>image_static_stylized_suburb_0082</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera move right. A stylized forest with blocky trees, distant hills, and grass.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout stays crisp and well rendered; the lower-scoring rollout is visibly washed out and texture-poor.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_suburb_0082">
          <source src="../../../assets/metric_examples/curated_20260801/941cad1029b1__image_static_stylized_suburb_0082__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7881</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_suburb_0082">
          <source src="../../../assets/metric_examples/curated_20260801/7e8f29555362__image_static_stylized_suburb_0082__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2531</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
