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
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和完全一致的展示 prompt，然后比较模型生成结果。paired 展示必须满足 input image 相同、prompt 文本相同，并且一个模型高分、一个模型低分。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `sharpness_retention`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/564b5211b4ae__image_static_photorealistic_passageways_0077.jpg" alt="输入图像: image_static_photorealistic_passageways_0077" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_static_photorealistic_passageways_0077</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan left. A japanese pathway with torii gates, stone lantern, and fox statues. Camera move left. A japanese garden with small pond, stone bridge, and red foliage. Camera pull out. A japanese garden with tea house, bamboo grove, and garden plants.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_passageways_0077">
      <source src="../../../assets/metric_examples/641521aff88e__cosmos3-nano-image-static__image_static_photorealistic_passageways_0077.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_passageways_0077">
      <source src="../../../assets/metric_examples/4e91b9f3e95c__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_passageways_0077.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2f79eec6bacb__image_static_photorealistic_public_spaces_0037.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0037" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_static_photorealistic_public_spaces_0037</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move right. Modern buildings with street and pedestrians.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_public_spaces_0037">
      <source src="../../../assets/metric_examples/7e1b20e236f8__cosmos3-nano-image-static__image_static_photorealistic_public_spaces_0037.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_public_spaces_0037">
      <source src="../../../assets/metric_examples/5b5643c5c871__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_public_spaces_0037.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/45fb3a6472c9__image_static_photorealistic_suburb_0012.jpg" alt="输入图像: image_static_photorealistic_suburb_0012" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_static_photorealistic_suburb_0012</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pull out. A suburban landscape with city skyline, green hills, and road. Camera move right. A suburban landscape with road, residential area, and green hills. Camera pan right. A residential area with park, playground, and green hills.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_suburb_0012">
      <source src="../../../assets/metric_examples/3c1ba8c5ffad__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_suburb_0012.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_suburb_0012">
      <source src="../../../assets/metric_examples/9f1698417770__dreamx_world__image_static_photorealistic_suburb_0012.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f116c80bb018__image_static_stylized_dining_spaces_0027.jpg" alt="输入图像: image_static_stylized_dining_spaces_0027" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_static_stylized_dining_spaces_0027</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move left. A modern kitchen with dining table, chairs, and greenery outside. Camera pan left. A modern kitchen with sofa, coffee table, and living area. Camera pull out. A modern kitchen with hallway, artwork, and indoor plants.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_dining_spaces_0027">
      <source src="../../../assets/metric_examples/166bdb8a9374__astra__image_static_stylized_dining_spaces_0027.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_dining_spaces_0027">
      <source src="../../../assets/metric_examples/8119a3b3dbfc__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_dining_spaces_0027.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2c0214795c05__image_static_stylized_passageways_0073.png" alt="输入图像: image_static_stylized_passageways_0073" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_static_stylized_passageways_0073</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera orbit left. A stylized hallway with console table, circular mirror, and lamps.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_passageways_0073">
      <source src="../../../assets/metric_examples/95936cc0040a__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_passageways_0073.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_passageways_0073">
      <source src="../../../assets/metric_examples/da49df4a04bd__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_passageways_0073.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/38fa6cdab0e6__image_static_stylized_public_spaces_0033.jpg" alt="输入图像: image_static_stylized_public_spaces_0033" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_static_stylized_public_spaces_0033</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera orbit right. A library hall with bookshelves, chandeliers, and reading tables.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_public_spaces_0033">
      <source src="../../../assets/metric_examples/d463b04eb239__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_public_spaces_0033.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_public_spaces_0033">
      <source src="../../../assets/metric_examples/7fd273eb810c__hy_worldplay_ar_rl_image_test_balanced_5000_5s__image_static_stylized_public_spaces_0033.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
