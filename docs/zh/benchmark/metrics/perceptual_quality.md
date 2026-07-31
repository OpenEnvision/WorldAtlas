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
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/662196f81a7b__image_static_stylized_dining_spaces_0028__input.jpg" alt="输入图像: image_static_stylized_dining_spaces_0028" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.3476</div>
      <h4><code>image_static_stylized_dining_spaces_0028</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pull out. A dining area with dining table, chairs, and place settings.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果保持干净、写实；较低分结果存在严重撕裂与重建伪影。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_dining_spaces_0028">
          <source src="../../../assets/metric_examples/curated_20260801/c55fe27825e7__image_static_stylized_dining_spaces_0028__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7350</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>perceptual_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_dining_spaces_0028">
          <source src="../../../assets/metric_examples/curated_20260801/2949a852d70a__image_static_stylized_dining_spaces_0028__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3874</div>
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
