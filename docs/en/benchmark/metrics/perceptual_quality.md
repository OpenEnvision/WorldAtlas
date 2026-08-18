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
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/1ae68213d599__image_static_photorealistic_aquatic_landscape_0079__input.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0079" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.3998</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0079</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. A stone archway with pathway, grassy area, and mountains.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_aquatic_landscape_0079">
          <source src="../../../assets/metric_examples/f47c73e21fbd__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0079.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7155</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>perceptual_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_aquatic_landscape_0079">
          <source src="../../../assets/metric_examples/faeffc40fff5__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_photorealistic_aquatic_landscape_0079.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3157</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>perceptual_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d774d136896a__image_static_stylized_aquatic_landscape_0110__input.jpg" alt="Input image: image_static_stylized_aquatic_landscape_0110" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.3910</div>
      <h4><code>image_static_stylized_aquatic_landscape_0110</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit right. An aquatic landscape with oil rig, industrial buildings, and rocky platform.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_aquatic_landscape_0110">
          <source src="../../../assets/metric_examples/0476470e1b85__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_aquatic_landscape_0110.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.6485</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>perceptual_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_aquatic_landscape_0110">
          <source src="../../../assets/metric_examples/dc4ffb2550a8__matrix_game2_5s__image_static_stylized_aquatic_landscape_0110.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2576</div>
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
