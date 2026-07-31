# 背景一致性（Background Consistency）

背景一致性度量生成视频中**场景外观的时序稳定性**。相邻帧之间若背景布局、纹理或全局场景上下文发生大幅跳变，得分会降低。

对解码帧 $x_i$，CLIP ViT-B/32 提取 L2 归一化嵌入 $f_i$。指标对相邻帧对余弦相似度取平均：

\[
S_{\mathrm{bg}}
=
\frac{1}{N-1}\sum_{i=1}^{N-1}\cos(f_i, f_{i+1})
\tag{1}
\]

其中 $N$ 为评测帧数，$\cos(\cdot,\cdot)$ 为 CLIP 图像特征之间的余弦相似度。WorldArena 将 $S_{\mathrm{bg}}$ 截断至 $[0, 1]$；得分越高表示背景越稳定。该指标**仅需 prediction**，且至少需要两帧。

## 参考文献 {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d0f020e9108d__image_static_stylized_aquatic_landscape_0063__input.jpg" alt="输入图像: image_static_stylized_aquatic_landscape_0063" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.1793</div>
      <h4><code>image_static_stylized_aquatic_landscape_0063</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move right. A stylized lake with trees, hills, and clouds.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果在镜头移动中保持湖面、山体与岸线连贯；较低分结果逐步丢失背景，末段退化为崩坏纹理。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_aquatic_landscape_0063">
          <source src="../../../assets/metric_examples/curated_20260801/a285b51b62d0__image_static_stylized_aquatic_landscape_0063__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9933</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>background_consistency</span><span>平均相邻帧相似度: 0.9933</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_aquatic_landscape_0063">
          <source src="../../../assets/metric_examples/curated_20260801/85e93576194e__image_static_stylized_aquatic_landscape_0063__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.8140</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>background_consistency</span><span>平均相邻帧相似度: 0.8140</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
