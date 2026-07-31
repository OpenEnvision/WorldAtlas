# Sharpness Retention

本指标通过监测边缘梯度的演化，评估生成视频中**细节稳定性**。成功产生的数值结果为 official：会进入 `quality` 维度和公开 leaderboard 向量。该实现采用向量化 Tenengrad 方法：分别计算水平方向（$G_x$）与竖直方向（$G_y$）Sobel 梯度的绝对值之和，构造二维清晰度向量以刻画纹理方向特征：

\[
\mathbf{g}_t
=
\begin{pmatrix}
\sum |G_x| \\
\sum |G_y|
\end{pmatrix}
\tag{6}
\]

其中 $G_x, G_y$ 为第 $t$ 帧灰度图经 $3\times3$ Sobel 算子得到的梯度响应。

引入**噪声感知熔断机制**以区分真实细节与高频伪影：记 $n_t$ 为第 $t$ 帧 BRISQUE 噪声分数，$\tau$ 为噪声阈值；当连续 $5$ 帧满足 $n_t > \tau$ 时，触发状态 $I_{\mathrm{trig}}=1$。融合惩罚逻辑 $M$ 将纹理相似度与熔断状态结合（相似度 $S(\cdot,\cdot)$ 采用余弦内积，定义同式 (3)）：

BRISQUE 对零方差帧没有定义。实现会给这类帧赋予配置中的最差值 `brisque_bad`，并继续把它计入聚合，因此纯色或黑帧坍缩会受到惩罚，而不会作为缺失分数被静默排除。

\[
M(x_t)
=
\begin{cases}
S(\mathbf{g}_t, \mathbf{g}_1), & I_{\mathrm{trig}} = 0 \;\wedge\; n_t < \tau \\[4pt]
\mathrm{clip}\!\bigl(1 - S(\mathbf{g}_t, \mathbf{g}_1),\, 0,\, 0.2\bigr), & \text{otherwise}
\end{cases}
\tag{11}
\]

对低–中分段细节增强，采用**上凸对数变换** $L(\cdot)$（$x \in [0,1]$，避免无意义对数并归一化）：

\[
L(x)
=
\frac{\ln(1 + k \cdot x)}{\ln(1 + k)},
\quad k > 0
\tag{10}
\]

其中 $k$ 控制上凸程度，视觉领域常用 $k \in [10, 20]$；分子保证 $x=0 \Rightarrow L(x)=0$，分母保证 $x=1 \Rightarrow L(x)=1$。

时序聚合采用归一化指数衰减权重 $\hat{w}(d)$（定义同亮度一致性指标式 (5)，$d \in [1, T-1]$ 为帧距）。综合得分为：

\[
S_{\mathrm{Sharpness}}
=
\sum_{d=1}^{T-1}
\hat{w}(d) \cdot L\!\left(M(x_{d+1})\right)
\tag{12}
\]

其中 $S(\mathbf{g}_t, \mathbf{g}_1)$ 为第 $t$ 帧与首帧清晰度向量的余弦相似度。该机制可在区分真实细节与高频噪声的同时，对系统性视觉崩塌施加惩罚。

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/0d2acb3aad70__image_static_stylized_city_0023__input.jpg" alt="输入图像: image_static_stylized_city_0023" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.9009</div>
      <h4><code>image_static_stylized_city_0023</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move right. A stylized city street with buildings, pedestrians, and trees. Camera pan right. A stylized city street with park gates, benches, and trees. Camera pull out. A stylized city street with fountain, people, and paved square.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果保留了清晰边缘和细部结构；较低分结果中的城市画面逐步模糊、溶解。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_city_0023">
          <source src="../../../assets/metric_examples/curated_20260801/2cfbdda4e56c__image_static_stylized_city_0023__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9858</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>sharpness_retention</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_city_0023">
          <source src="../../../assets/metric_examples/curated_20260801/b4f81c8fb084__image_static_stylized_city_0023__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0849</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>sharpness_retention</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
