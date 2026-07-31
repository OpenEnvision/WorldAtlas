# 亮度一致性

本指标用于量化视频序列中**各帧与初始帧之间亮度分布的时序一致性**，以应对帧间亮度偏移与突变导致的视觉稳定性不足。与仅比较全局平均亮度的标量指标不同，Brightness Consistency 保留暗/中/亮三档像素占比，从而对曝光分布形状的变化更敏感。

WorldArena 将本算法作为正式指标 **`brightness_distribution_consistency`**。旧的标量别名 `brightness_consistency` 仍为 diagnostic，不进入七项正式 quality 总分。

### 三档亮度分布向量

将每帧灰度像素划分为 **暗（dark）**、**中（mid）**、**亮（bright）** 三档，构造三维亮度分布向量：

\[
\mathbf{v}_t
=
\begin{pmatrix}
p_{\mathrm{dark}} \\
p_{\mathrm{mid}} \\
p_{\mathrm{bright}}
\end{pmatrix}
\]

其中 $p_{\mathrm{dark}}$、$p_{\mathrm{bright}}$ 分别为灰度低于暗部阈值、不低于亮部阈值的像素占比，$p_{\mathrm{mid}} = 1 - p_{\mathrm{dark}} - p_{\mathrm{bright}}$。WorldArena 默认 `dark_threshold=85`、`bright_threshold=170`（OpenCV 灰度 $[0,255]$）。

### 相似度与修正 Softmax

第 $t$ 帧（$t \geq 2$）相对首帧 $\mathbf{v}_1$ 的分布相似度采用余弦内积：

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

对单样本分数 $x \in [0,1]$ 施加修正 Softmax 变换（增强高分段区分度，$\lambda > 0$）：

\[
T(x)
=
\frac{e^{\lambda x} - 1}{e^{\lambda} - 1}
\tag{4}
\]

默认 $\lambda = 4.0$（`softmax_lambda`）。

### 时序加权聚合

帧距 $d \in [1, T-1]$（$d=1$ 对应第 $2$ 帧）采用归一化指数衰减权重（$\alpha > 0$）：

\[
w(d) = e^{-\alpha \cdot d},
\qquad
\hat{w}(d)
=
\frac{w(d)}{\sum_{k=1}^{T-1} w(k)}
\tag{5}
\]

默认 $\alpha = 0.05$（`temporal_decay`）。综合亮度一致性得分为：

\[
S_{\mathrm{Brightness}}
=
\sum_{d=1}^{T-1}
\hat{w}(d) \cdot T\!\left(S(\mathbf{v}_{d+1}, \mathbf{v}_1)\right)
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
      <img src="../../../assets/metric_examples/inputs/38c5df592fc3__image_static_photorealistic_aquatic_landscape_0098.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0098" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.9727</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0098</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan right. An aquatic landscape with pebble beach, calm sea, and blue sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果持续保持明亮海岸的曝光分布；较低分结果很快转成昏暗、浑浊的亮度分布。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_aquatic_landscape_0098">
          <source src="../../../assets/metric_examples/6dcb1b7c126c__cosmos3-nano-image-static__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9991</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_aquatic_landscape_0098">
          <source src="../../../assets/metric_examples/curated_20260801/f1f662bdef18__image_static_photorealistic_aquatic_landscape_0098__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0264</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>brightness_distribution_consistency</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "WorldAtlas Arena — Brightness Consistency metric family"

    Three-bin brightness distribution vectors with cosine similarity to the initial frame, modified Softmax enhancement, and normalized exponential temporal weighting.
