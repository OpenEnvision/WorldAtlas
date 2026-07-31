# Style Consistency

本指标用于评估生成 rollout 内视觉风格（色彩、纹理与粗粒度外观统计）的跨场景稳定性。给定按评测协议抽取的 anchor 帧序列 $\{I_t\}_{t=1}^{T}$（$T \geq 2$），以首帧 $I_1$ 为风格参考，将后续各幕 anchor $I_{d+1}$（$d \in [1, T-1]$）与 $I_1$ 在 VGG19 多层 Gram 表征上对齐比较；**不使用** ground-truth reference 参与打分。

对每帧 $I_t$，先经 resize（CUDA 上为 $512 \times 512$，CPU 上为 $128 \times 128$）、`ToTensor` 及 ImageNet 均值–方差归一化，再送入 VGG19 `features` 前向传播。在卷积层 $\mathrm{conv}_1, \ldots, \mathrm{conv}_5$ 的输出处（ReLU 之前）提取特征图 $F_{t,l} \in \mathbb{R}^{C \times H \times W}$。Gram 矩阵定义为

\[
G(F)
=
\frac{
    \mathrm{vec}(F)\,\mathrm{vec}(F)^{\top}
}{
    C \cdot H \cdot W
}
\tag{1}
\]

其中 $\mathrm{vec}(\cdot)$ 为向量化算子。帧 $d+1$ 在层 $l$ 上的风格距离为

\[
\delta_{d,l}
=
\mathrm{MSE}\!\left(
    G(F_{d+1,l}),\,
    G(F_{1,l})
\right)
\tag{2}
\]

单帧距离为五层距离之和：

\[
d_d
=
\sum_{l \in \{\mathrm{conv}_1, \ldots, \mathrm{conv}_5\}}
\delta_{d,l}
\tag{3}
\]

原始分数取后续 anchor 帧距离的算术平均（**越低越好**）：

\[
\mathrm{raw}
=
\frac{1}{T-1}
\sum_{d=1}^{T-1} d_d
\tag{4}
\]

归一化采用配置的经验界 $\varepsilon_{\min}=0$、$\varepsilon_{\max}=0.007$（`higher_is_better=false`）。先将 raw 钳位至经验区间，再映射为 $[0,1]$ 且**越高越好**的归一化分数：

\[
\mathrm{raw}'
=
\mathrm{clip}(\mathrm{raw},\, \varepsilon_{\min},\, \varepsilon_{\max})
\tag{5}
\]

\[
S_{\mathrm{style}}
=
1
-
\frac{
    \mathrm{raw}' - \varepsilon_{\min}
}{
    \varepsilon_{\max} - \varepsilon_{\min}
}
\tag{6}
\]

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/f7d43b5c2adb__image_static_photorealistic_passageways_0144__input.jpg" alt="输入图像: image_static_photorealistic_passageways_0144" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.0103</div>
      <h4><code>image_static_photorealistic_passageways_0144</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan left. A modern hallway with geometric walls, adjacent corridor, and metallic ceiling lights.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果保持连贯的写实几何室内风格；较低分结果则切换成无关的反光条纹外观。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_passageways_0144">
          <source src="../../../assets/metric_examples/curated_20260801/e33a33498b58__image_static_photorealistic_passageways_0144__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9999</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>style_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_passageways_0144">
          <source src="../../../assets/metric_examples/curated_20260801/a5ac53bfecb4__image_static_photorealistic_passageways_0144__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.9896</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>style_consistency</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "Very Deep Convolutional Networks for Large-Scale Image Recognition"

    Karen Simonyan and Andrew Zisserman.
    arXiv:1409.1556, 2014.
    [https://arxiv.org/pdf/1409.1556](https://arxiv.org/pdf/1409.1556)
