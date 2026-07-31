# Prompt Alignment

Prompt Alignment measures **semantic agreement** between protocol anchor frames of a generated video and the target instruction \(T = \texttt{sample.prompt\_target}\). It evaluates whether the prediction realizes the requested scene content at benchmark-sampled time points, not whether the full rollout is temporally coherent or visually realistic.

Let \(\mathcal{A} = \{f_i\}_{i=1}^{M}\) denote the anchor-frame set selected by the evaluation protocol from the decoded prediction. With CLIP ViT-B/16 image encoder \(\phi_I\) and text encoder \(\phi_T\), define L2-normalized embeddings \(\mathbf{e}_i = \phi_I(f_i)\) and \(\mathbf{t} = \phi_T(T)\). The native CLIPScore for frame \(i\) is

\[
s_i = \max\!\left(100 \cdot \langle \mathbf{e}_i, \mathbf{t} \rangle,\, 0\right)
\tag{1}
\]

WorldArena maps each anchor score to \([0,1]\) via \(\tilde{s}_i = \mathrm{clamp}(s_i / 100,\, 0,\, 1)\) and reports the mean

\[
S_{\mathrm{prompt}} = \frac{1}{M}\sum_{i=1}^{M} \tilde{s}_i
\tag{2}
\]

Higher \(S_{\mathrm{prompt}}\) indicates stronger instruction following at protocol anchors. The metric is **official** when explicit instruction supervision is available and conditioning is not self-referential (`conditioning_strategy ≠ self`). Image suites may condition on a single reference frame, but the model prediction must still be a video.

The examples below are curated from the complete 2026-07-17 `image_static` snapshot: 20 models evaluated on the same 3,372 cases. Every side-by-side pair is strictly verified to share the sample ID, conditioning-image SHA-256, and actual generation-prompt SHA-256; both generation records must also identify that image as the conditioning input. “Good” and “poor” refer only to Prompt Alignment—not overall video quality—and each score is the mean over four `segment_random` protocol anchors.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/49bd12360acb__image_static_stylized_living_spaces_0053.jpg" alt="Input image: image_static_stylized_living_spaces_0053" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.1824</div>
      <h4><code>image_static_stylized_living_spaces_0053</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera push in. An attic bedroom with tree tops, sunlight, and large window.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout realizes the requested attic bedroom and large window; the lower-scoring rollout drifts into outdoor and abstract content.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0053">
          <source src="../../../assets/metric_examples/curated_20260801/28be3603d44d__image_static_stylized_living_spaces_0053__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.3396</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>prompt_alignment</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0053">
          <source src="../../../assets/metric_examples/e1dae4a903b7__astra_image_static__image_static_stylized_living_spaces_0053.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.1572</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>prompt_alignment</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
