# Color Temperature Consistency

本指标用于评估生成视频中**环境氛围（色温逻辑）**的一致性。在 HSV 色彩空间中分析 Hue 维度：将 hue 取值范围 $[0, 179]$ 划分为 $7$ 个核心区间，构造 hue 特征向量 $\mathbf{h}_t \in \mathbb{R}^{7}$；再计算整段序列相对首帧的加权感知相似度。为平衡**全局恒定**与**局部过渡**，第 $t$ 帧（$t \geq 2$）的相似度定义为「当前帧–首帧」与「当前帧–前一帧」余弦相似度的算术平均。相似度 $S(\cdot, \cdot)$ 采用余弦内积（定义同亮度分布一致性指标式 (3)）：

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

\[
\bar{S}_t
=
\frac{
    S(\mathbf{h}_t, \mathbf{h}_1)
    +
    S(\mathbf{h}_t, \mathbf{h}_{t-1})
}{
    2
}
\tag{7}
\]

对每帧 $I_t$，先将 RGB 转为 HSV，以饱和度 $S$ 为权重在 $7$ 个 hue 区间上统计直方图并归一化，得到 $\mathbf{h}_t$（WorldArena 默认 `hue_bins=7`，区间 $[0, 180)$）。

最终分数采用**归一化指数衰减加权求和**，权重 $\hat{w}'(d)$ 为改进版指数核（相较亮度指标 $\hat{w}(d)$ 衰减更快，$\beta > \alpha$），并对相似度施加修正 Softmax 变换 $T(\cdot)$（定义同式 (4)，默认 $\lambda = 4.0$）：

\[
T(x)
=
\frac{e^{\lambda x} - 1}{e^{\lambda} - 1},
\quad \lambda > 0
\tag{4}
\]

\[
w'(d) = e^{-\beta \cdot d},
\qquad
\hat{w}'(d)
=
\frac{w'(d)}{\sum_{k=1}^{T-1} w'(k)}
\quad (\beta > 0)
\tag{8}
\]

其中 $d \in [1, T-1]$ 为帧距（$d=1$ 对应第 $2$ 帧相对首帧），$\beta$ 为衰减系数（越大则远端帧权重衰减越快）。默认配置下 $\beta = 0.10$，亮度分布指标中取 $\alpha = 0.05$，满足 $\beta > \alpha$，从而对长视频常见的「色漂」现象施加更强惩罚，保证场景色温逻辑一致。综合得分为：

\[
S_{\mathrm{Color}}
=
\sum_{d=1}^{T-1}
\hat{w}'(d) \cdot T\!\left(\bar{S}_{d+1}\right)
\tag{9}
\]

WorldArena 中本算法对应正式指标 **`color_temperature_constraint`**（`hue_distribution` 后端）；归一化配置下 raw 与 normalized 均为 $[0,1]$，**越高越好**。旧的标量别名 `color_temperature_consistency` 仍为 diagnostic，不进入七项正式 quality 总分。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/27fe1e1e874e__image_static_stylized_aquatic_landscape_0098__input.jpg" alt="输入图像: image_static_stylized_aquatic_landscape_0098" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8068</div>
      <h4><code>image_static_stylized_aquatic_landscape_0098</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan left. An aquatic landscape with beach, cliffs, and ocean.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_aquatic_landscape_0098">
          <source src="../../../assets/metric_examples/curated_20260801/e34a878acd25__image_static_stylized_aquatic_landscape_0098__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9983</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_aquatic_landscape_0098">
          <source src="../../../assets/metric_examples/curated_20260801/23de9c97cbb5__image_static_stylized_aquatic_landscape_0098__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1915</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/cb824c72b1ab__image_static_stylized_workspaces_0088__input.jpg" alt="输入图像: image_static_stylized_workspaces_0088" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8132</div>
      <h4><code>image_static_stylized_workspaces_0088</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A modern conference room with couch, coffee table, and bookshelves.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_workspaces_0088">
          <source src="../../../assets/metric_examples/curated_20260801/270f2c11486a__image_static_stylized_workspaces_0088__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9998</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_workspaces_0088">
          <source src="../../../assets/metric_examples/curated_20260801/b3fb040b2633__image_static_stylized_workspaces_0088__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1866</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/d2f91397bbee__image_static_stylized_suburb_0132__input.jpg" alt="输入图像: image_static_stylized_suburb_0132" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.6806</div>
      <h4><code>image_static_stylized_suburb_0132</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A suburban street with houses, trees, and road.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_suburb_0132">
          <source src="../../../assets/metric_examples/curated_20260801/a8b3af0667b6__image_static_stylized_suburb_0132__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9968</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_suburb_0132">
          <source src="../../../assets/metric_examples/curated_20260801/9c8023a7cd72__image_static_stylized_suburb_0132__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3162</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>color_temperature_constraint</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
