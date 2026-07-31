# Optical Flow AEPE

Optical Flow AEPE quantifies **short-horizon photometric consistency** in a generated video via forward–backward optical-flow round trips. It measures whether local appearance remains stable across consecutive frames—not whether motion matches ground truth.

For adjacent frames \(A\) and \(B\), sample points \(p_{A,i}\) in a center crop. Forward flow \(F_{A\to B}\) tracks each point to \(p_{B,i} = p_{A,i} + F_{A\to B}(p_{A,i})\); backward flow returns \(p'_{A,i} = p_{B,i} + F_{B\to A}(p_{B,i})\). The per-pair average end-point error is

\[
e_{\mathrm{AEPE}} = \frac{1}{N}\sum_{i=1}^{N} \left\| p_{A,i} - p'_{A,i} \right\|_2
\tag{1}
\]

The raw score averages \(e_{\mathrm{AEPE}}\) over all consecutive frame pairs. WorldArena reverse-normalizes with empirical range \([0, 1.192]\). Lower error (higher normalized score) indicates more stable local appearance. The metric is **official** when SEA-RAFT computation succeeds.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/ba2f269e3cf8__image_static_stylized_public_spaces_0139__input.jpg" alt="Input image: image_static_stylized_public_spaces_0139" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.9562</div>
      <h4><code>image_static_stylized_public_spaces_0139</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera orbit right. A grand atrium with model city, sculptures, and informational desks.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring orbit changes viewpoint while preserving a coherent atrium; the lower-scoring rollout replaces the structure, producing flow inconsistent with the intended camera motion.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_public_spaces_0139">
          <source src="../../../assets/metric_examples/curated_20260801/c52469ee801b__image_static_stylized_public_spaces_0139__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9655</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>optical_flow_aepe</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_public_spaces_0139">
          <source src="../../../assets/metric_examples/curated_20260801/49445d116aba__image_static_stylized_public_spaces_0139__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0093</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>optical_flow_aepe</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., ECCV 2024.
    [https://arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
