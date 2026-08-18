# Optical Flow AEPE

Optical Flow AEPE quantifies **short-horizon photometric consistency** in a generated video via forward–backward optical-flow round trips. It measures whether local appearance remains stable across consecutive frames—not whether motion matches ground truth.

For adjacent frames \(A\) and \(B\), sample points \(p_{A,i}\) in a center crop. Forward flow \(F_{A\to B}\) tracks each point to \(p_{B,i} = p_{A,i} + F_{A\to B}(p_{A,i})\); backward flow returns \(p'_{A,i} = p_{B,i} + F_{B\to A}(p_{B,i})\). The per-pair average end-point error is

\[
e_{\mathrm{AEPE}} = \frac{1}{N}\sum_{i=1}^{N} \left\| p_{A,i} - p'_{A,i} \right\|_2
\tag{1}
\]

The raw score averages \(e_{\mathrm{AEPE}}\) over all consecutive frame pairs. WorldArena reverse-normalizes with empirical range \([0, 1.192]\). Lower error (higher normalized score) indicates more stable local appearance. The metric is **official** when SEA-RAFT computation succeeds.

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/357ac0fd24a0__image_static_stylized_city_0098__input.jpg" alt="Input image: image_static_stylized_city_0098" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9399</div>
      <h4><code>image_static_stylized_city_0098</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera push in. An urban street with tall tower, building facades, and windows.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_city_0098">
          <source src="../../../assets/metric_examples/curated_20260801/940d25c65988__image_static_stylized_city_0098__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9566</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>optical_flow_aepe</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_city_0098">
          <source src="../../../assets/metric_examples/curated_20260801/d93109cae860__image_static_stylized_city_0098__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0167</div>
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
