# Color Temperature Consistency

This metric evaluates the consistency of **environmental atmosphere (color-temperature logic)** in generated videos. We analyze the Hue channel in HSV color space: the hue spectrum $[0, 179]$ is divided into $7$ core intervals to construct a hue feature vector $\mathbf{h}_t \in \mathbb{R}^{7}$. We then compute a weighted perceptual similarity of the full sequence relative to the initial frame. To balance **global constancy** and **local transition**, the similarity at frame $t$ ($t \geq 2$) is the arithmetic mean of current-to-first and current-to-previous similarities. The similarity $S(\cdot, \cdot)$ follows the cosine inner product (same as Eq. (3) in the brightness-distribution metric):

\[
S(a, b)
=
\frac{a \cdot b}{\|a\|\,\|b\|}
=
\frac{\sum_{i=1}^{n} a_i b_i}{
    \sqrt{\sum_{i=1}^{n} a_i^2}\,\sqrt{\sum_{i=1}^{n} b_i^2}
}
\tag{3}
\]

\[
\bar{S}_t
=
\frac{
    S(\mathbf{h}_t, \mathbf{h}_1)
    +
    S(\mathbf{h}_t, \mathbf{h}_{t-1})
}{
    2
}
\tag{7}
\]

For each frame $I_t$, RGB is converted to HSV; a saturation-weighted histogram over $7$ hue bins on $[0, 180)$ is normalized to form $\mathbf{h}_t$ (WorldArena default `hue_bins=7`).

The final score uses a **normalized exponentially decaying weighted sum**, where weights $\hat{w}'(d)$ are an improved variant of $\hat{w}(d)$ with faster decay ($\beta > \alpha$), and similarities are passed through the modified Softmax transform $T(\cdot)$ (Eq. (4), default $\lambda = 4.0$):

\[
T(x)
=
\frac{e^{\lambda x} - 1}{e^{\lambda} - 1},
\quad \lambda > 0
\tag{4}
\]

\[
w'(d) = e^{-\beta \cdot d},
\qquad
\hat{w}'(d)
=
\frac{w'(d)}{\sum_{k=1}^{T-1} w'(k)}
\quad (\beta > 0)
\tag{8}
\]

Here $d \in [1, T-1]$ is the frame distance ($d=1$ for the second frame relative to the first), and $\beta$ is the decay coefficient. The default $\beta = 0.10$ exceeds the brightness metric's $\alpha = 0.05$, so $\hat{w}'(d)$ decays faster than $\hat{w}(d)$ and long-video color drift is penalized more strongly. The comprehensive score is:

\[
S_{\mathrm{Color}}
=
\sum_{d=1}^{T-1}
\hat{w}'(d) \cdot T\!\left(\bar{S}_{d+1}\right)
\tag{9}
\]

In WorldArena this algorithm is implemented as the official metric **`color_temperature_constraint`** (`hue_distribution` backend). Under the default normalization config, raw and normalized outputs lie in $[0,1]$ with **higher is better**. The legacy scalar alias `color_temperature_consistency` remains diagnostic and is not part of the seven-metric official quality total.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/41bd11d5eb2a__image_static_stylized_passageways_0041.jpg" alt="Input image: image_static_stylized_passageways_0041" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.9030</div>
      <h4><code>image_static_stylized_passageways_0041</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera orbit left. A stylized courtyard with arches, columns, and round table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout preserves a coherent warm palette; the lower-scoring rollout drifts into a desaturated gray appearance.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_passageways_0041">
          <source src="../../../assets/metric_examples/4203fbe1e26a__cameractrl__image_static_stylized_passageways_0041.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9994</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_passageways_0041">
          <source src="../../../assets/metric_examples/5063767b647a__astra__image_static_stylized_passageways_0041.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0964</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
