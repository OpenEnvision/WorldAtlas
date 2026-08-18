# Perceptual Quality

**perceptual_quality** 衡量预测结果**自身的无参考视觉质量**，不将 prediction 与 GT reference 做逐帧配对比较。它关注锚点帧层面的锐度、干净程度与审美合理性，**不**评估语义正确性、prompt 对齐或几何/物理合规性。

WorldArena 将两类互补的无参考 IQA 信号融合为单一感知质量分数：CLIP-IQA+ 刻画整体视觉自然度与感知协调性；LAION 美学预测器刻画审美吸引力。二者均在 prediction anchor 帧上独立打分后再聚合。

### CLIP-IQA+ 组件

CLIP-IQA+ 基于 CLIP 视觉—语言表征评估单帧图像的整体感知质量。对锚点帧 $I_i$，模型输出 native 分数 $q_i^{\mathrm{native}}$，归一化为：

\[
s_i^{\mathrm{iqa}}
=
\mathrm{clip}\!\left(\frac{q_i^{\mathrm{native}}}{1},\, 0,\, 1\right)
\tag{1}
\]

### 美学组件

美学分数沿用 CLIP ViT-L/14 特征与 LAION-Aesthetics 线性回归头（pyiqa `laion_aes`）。对锚点帧 $I_i$：

\[
s_i^{\mathrm{aes}}
=
\mathrm{clip}\!\left(\frac{g(\mathrm{norm}(\mathrm{CLIP}(I_i)))}{10},\, 0,\, 1\right)
\tag{2}
\]

### 帧级与视频级聚合

每帧感知质量取两组件的算术平均：

\[
s_i
=
\frac{s_i^{\mathrm{iqa}} + s_i^{\mathrm{aes}}}{2}
\tag{3}
\]

设协议选出 $N$ 个 prediction anchor 帧，指标 raw 值为：

\[
S_{\mathrm{pq}}
=
\frac{1}{N}\sum_{i=1}^{N} s_i
\tag{4}
\]

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/1ae68213d599__image_static_photorealistic_aquatic_landscape_0079__input.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0079" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.3998</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0079</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A stone archway with pathway, grassy area, and mountains.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_aquatic_landscape_0079">
          <source src="../../../assets/metric_examples/f47c73e21fbd__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0079.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7155</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>perceptual_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_aquatic_landscape_0079">
          <source src="../../../assets/metric_examples/faeffc40fff5__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_photorealistic_aquatic_landscape_0079.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3157</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>perceptual_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d774d136896a__image_static_stylized_aquatic_landscape_0110__input.jpg" alt="输入图像: image_static_stylized_aquatic_landscape_0110" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.3910</div>
      <h4><code>image_static_stylized_aquatic_landscape_0110</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit right. An aquatic landscape with oil rig, industrial buildings, and rocky platform.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_aquatic_landscape_0110">
          <source src="../../../assets/metric_examples/0476470e1b85__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_aquatic_landscape_0110.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.6485</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>perceptual_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_aquatic_landscape_0110">
          <source src="../../../assets/metric_examples/dc4ffb2550a8__matrix_game2_5s__image_static_stylized_aquatic_landscape_0110.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.2576</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>perceptual_quality</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

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
