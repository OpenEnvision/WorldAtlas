# Perceptual Quality

**perceptual_quality** measures the **no-reference visual quality of the prediction itself**, without pairing prediction frames against a ground-truth reference rollout. It evaluates sharpness, cleanliness, and aesthetic plausibility at prediction anchor frames. It does **not** measure semantic correctness, prompt alignment, or geometric/physical compliance.

WorldArena fuses two complementary no-reference IQA signals into one perceptual score: **CLIP-IQA+** captures overall visual naturalness and perceptual coherence, while the **LAION aesthetic predictor** captures aesthetic appeal. Both are scored independently on prediction anchors and then aggregated.

### CLIP-IQA+ Component

CLIP-IQA+ assesses holistic perceptual quality from CLIP visual–language representations. For anchor frame $I_i$, the model outputs a native score $q_i^{\mathrm{native}}$, normalized as:

\[
s_i^{\mathrm{iqa}}
=
\mathrm{clip}\!\left(\frac{q_i^{\mathrm{native}}}{1},\, 0,\, 1\right)
\tag{1}
\]

### Aesthetic Component

The aesthetic score uses CLIP ViT-L/14 features with a LAION-Aesthetics linear regressor (pyiqa `laion_aes`). For anchor frame $I_i$:

\[
s_i^{\mathrm{aes}}
=
\mathrm{clip}\!\left(\frac{g(\mathrm{norm}(\mathrm{CLIP}(I_i)))}{10},\, 0,\, 1\right)
\tag{2}
\]

### Frame-Level and Video-Level Aggregation

Per-frame perceptual quality is the arithmetic mean of both components:

\[
s_i
=
\frac{s_i^{\mathrm{iqa}} + s_i^{\mathrm{aes}}}{2}
\tag{3}
\]

Given $N$ protocol-selected prediction anchor frames, the raw metric is:

\[
S_{\mathrm{pq}}
=
\frac{1}{N}\sum_{i=1}^{N} s_i
\tag{4}
\]

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/662196f81a7b__image_static_stylized_dining_spaces_0028__input.jpg" alt="Input image: image_static_stylized_dining_spaces_0028" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.3476</div>
      <h4><code>image_static_stylized_dining_spaces_0028</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pull out. A dining area with dining table, chairs, and place settings.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout remains clean and photorealistic; the lower-scoring rollout contains severe tearing and reconstruction artifacts.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_dining_spaces_0028">
          <source src="../../../assets/metric_examples/curated_20260801/c55fe27825e7__image_static_stylized_dining_spaces_0028__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7350</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>perceptual_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_dining_spaces_0028">
          <source src="../../../assets/metric_examples/curated_20260801/2949a852d70a__image_static_stylized_dining_spaces_0028__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3874</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>perceptual_quality</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "Exploring CLIP for Assessing the Look and Feel of Images (CLIP-IQA+)"

    Jianyi Wang, Kelvin CK Chan, Chen Change Loy.
    AAAI 2023.
    [https://arxiv.org/abs/2207.12396](https://arxiv.org/abs/2207.12396)

!!! quote "LAION-Aesthetics Predictor"

    CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings; exposed via pyiqa `laion_aes`.

??? note "BibTeX — CLIP-IQA+"

    ```bibtex
    @inproceedings{wang2022exploring,
        author = {Wang, Jianyi and Chan, Kelvin CK and Loy, Chen Change},
        title = {Exploring CLIP for Assessing the Look and Feel of Images},
        booktitle = {AAAI},
        year = {2023}
    }
    ```
