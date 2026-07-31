# Motion Accuracy

Motion Accuracy 衡量受监督动态区域内**前景运动显著性**。SAM2 将初始分割 mask 传播至整段预测；SEA-RAFT 估计相邻帧光流。该指标检验 mask 内物体区域是否比背景呈现更强运动——**不**将实际运动与目标轨迹、方向或速度曲线做严格对比。

对相邻帧对 \((I_t, I_{t+1})\)，设 \(\mathbf{u}_t\) 为光流场、\(M_t\) 为传播（膨胀后）物体 mask。定义幅值 \(m_t = \|\mathbf{u}_t\|_2\) 及

\[
\Delta_t = \max_{p \in M_t} m_t(p) - \max_{p \notin M_t} m_t(p)
\tag{1}
\]

原始得分为所有帧对的平均显著性：

\[
S_{\mathrm{mot}}^{\mathrm{raw}} = \frac{1}{N-1}\sum_{t=1}^{N-1} \Delta_t
\tag{2}
\]

WorldArena 以 \(\mu = -0.1965\)、\(\sigma = 2.3687\) 做 z-score 校准，裁剪至 \([-1.6147, 1.5180]\)，并映射至 \([0.25, 0.75]\)。归一化得分越高，表示相对背景的前景 mask 运动越显著。当 `sample.mask_path` 下存在逐帧分割 mask 时，该指标为 **official**。

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/ae595b686cab__image_dynamic_stylized_articulated_0020__input.jpg" alt="输入图像: image_dynamic_stylized_articulated_0020" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.8605</div>
      <h4><code>image_dynamic_stylized_articulated_0020</code></h4>
      <div class="metric-case-meta"><span>套件: image_dynamic</span><span>场景: stylized / articulated</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera fixed. Two stylized sharks glide over the underwater landscape, their tails sweeping gently as they pass seaweed and rocky formations. Sunlight filters through the clear water while the seabed remains steady.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果让两条鲨鱼成为相对稳定海床上的显著运动前景；较低分结果把物体形变与更大范围的场景运动混在一起。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_dynamic_stylized_articulated_0020">
          <source src="../../../assets/metric_examples/curated_20260801/ceb76b5f7507__image_dynamic_stylized_articulated_0020__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9535</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>motion_accuracy</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_dynamic_stylized_articulated_0020">
          <source src="../../../assets/metric_examples/curated_20260801/8f5a70ee480e__image_dynamic_stylized_articulated_0020__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0930</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>motion_accuracy</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "SAM 2 — Segment Anything in Images and Videos"

    Ravi et al., 2024.
    [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

!!! quote "SEA-RAFT — Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., 2024.
    [https://arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
