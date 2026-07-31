# Background Consistency

Background Consistency measures **temporal stability of scene appearance** in a generated video. Large shifts in background layout, texture, or global scene context between adjacent frames lower the score.

For each decoded frame $x_i$, CLIP ViT-B/32 extracts a normalized embedding $f_i$. The metric averages cosine similarity over consecutive frame pairs:

\[
S_{\mathrm{bg}}
=
\frac{1}{N-1}\sum_{i=1}^{N-1}\cos(f_i, f_{i+1})
\tag{1}
\]

where $N$ is the number of evaluated frames and $\cos(\cdot,\cdot)$ is cosine similarity between L2-normalized CLIP image features. WorldArena clamps $S_{\mathrm{bg}}$ to $[0, 1]$; higher values indicate more stable backgrounds. The metric is **prediction-only** and requires at least two frames.

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d0f020e9108d__image_static_stylized_aquatic_landscape_0063__input.jpg" alt="Input image: image_static_stylized_aquatic_landscape_0063" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.1793</div>
      <h4><code>image_static_stylized_aquatic_landscape_0063</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera move right. A stylized lake with trees, hills, and clouds.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout keeps the lake, mountain, and shoreline coherent through the camera move; the lower-scoring rollout loses the background and ends in a collapsed texture.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_aquatic_landscape_0063">
          <source src="../../../assets/metric_examples/curated_20260801/a285b51b62d0__image_static_stylized_aquatic_landscape_0063__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9933</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.9933</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_aquatic_landscape_0063">
          <source src="../../../assets/metric_examples/curated_20260801/85e93576194e__image_static_stylized_aquatic_landscape_0063__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.8140</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>background_consistency</span><span>Mean adjacent-frame similarity: 0.8140</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
