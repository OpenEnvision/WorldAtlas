# Optical Flow AEPE

Optical Flow AEPE 通过正反向光流往返量化生成视频的**短时 photometric 一致**。它衡量局部外观在相邻帧间是否稳定，而非运动是否与 ground truth 一致。

对相邻帧 \(A\)、\(B\)，在中心裁剪区采样点 \(p_{A,i}\)。正向流 \(F_{A\to B}\) 将其跟踪至 \(p_{B,i} = p_{A,i} + F_{A\to B}(p_{A,i})\)；反向流返回 \(p'_{A,i} = p_{B,i} + F_{B\to A}(p_{B,i})\)。帧对平均端点误差为

\[
e_{\mathrm{AEPE}} = \frac{1}{N}\sum_{i=1}^{N} \left\| p_{A,i} - p'_{A,i} \right\|_2
\tag{1}
\]

原始得分对所有相邻帧对的 \(e_{\mathrm{AEPE}}\) 取均值。WorldArena 在经验范围 \([0, 1.192]\) 上反向归一化。误差越低（归一化得分越高）表示局部外观越稳定。SEA-RAFT 计算成功时为 **official**。

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/ba2f269e3cf8__image_static_stylized_public_spaces_0139__input.jpg" alt="输入图像: image_static_stylized_public_spaces_0139" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.9562</div>
      <h4><code>image_static_stylized_public_spaces_0139</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit right. A grand atrium with model city, sculptures, and informational desks.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果在改变视角时仍保持中庭结构连贯；较低分结果替换了主体结构，光流与预期镜头运动不再一致。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_public_spaces_0139">
          <source src="../../../assets/metric_examples/curated_20260801/c52469ee801b__image_static_stylized_public_spaces_0139__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9655</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>optical_flow_aepe</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_public_spaces_0139">
          <source src="../../../assets/metric_examples/curated_20260801/49445d116aba__image_static_stylized_public_spaces_0139__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0093</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>optical_flow_aepe</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., ECCV 2024.
    [https://arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
