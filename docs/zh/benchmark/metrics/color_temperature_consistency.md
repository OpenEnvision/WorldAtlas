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
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和完全一致的展示 prompt，然后比较模型生成结果。paired 展示必须满足 input image 相同、prompt 文本相同，并且一个模型高分、一个模型低分。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `color_temperature_constraint`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/92f03c71a628__image_static_photorealistic_living_spaces_0088.jpg" alt="输入图像: image_static_photorealistic_living_spaces_0088" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 0.9032</div>
      <h4><code>image_static_photorealistic_living_spaces_0088</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move right. A living room with sofa, dining table, and mirror. Camera pan right. A dining area with dining table, chairs, and window. Camera pull out. A window view with plants and outdoor scenery.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_living_spaces_0088">
      <source src="../../../assets/metric_examples/527e53ab225d__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_living_spaces_0088.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_living_spaces_0088">
      <source src="../../../assets/metric_examples/3c83fb5cbc12__cosmos3-nano-image-static-8gpu-singleload-20260612_170108__image_static_photorealistic_living_spaces_0088.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0968</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/41bd11d5eb2a__image_static_stylized_passageways_0041.jpg" alt="输入图像: image_static_stylized_passageways_0041" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同一输入图像 + 同一 prompt · 分差 0.9030</div>
      <h4><code>image_static_stylized_passageways_0041</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera orbit left. A stylized courtyard with arches, columns, and round table.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_passageways_0041">
      <source src="../../../assets/metric_examples/4203fbe1e26a__cameractrl__image_static_stylized_passageways_0041.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9994</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_passageways_0041">
      <source src="../../../assets/metric_examples/5063767b647a__astra__image_static_stylized_passageways_0041.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0964</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/0e292106cb09__image_static_photorealistic_dining_spaces_0161.jpg" alt="输入图像: image_static_photorealistic_dining_spaces_0161" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同一输入图像 + 同一 prompt · 分差 0.8989</div>
      <h4><code>image_static_photorealistic_dining_spaces_0161</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan left. A dilapidated kitchen with wheelchair, dim lighting, and peeling wallpaper.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_dining_spaces_0161">
      <source src="../../../assets/metric_examples/267ff2365de7__dreamx_world__image_static_photorealistic_dining_spaces_0161.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_dining_spaces_0161">
      <source src="../../../assets/metric_examples/d9bbad493d2b__hy_worldplay_ar_rl_image_test_balanced_5000_5s__image_static_photorealistic_dining_spaces_0161.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.1011</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/9f9df2c4631a__image_static_stylized_dining_spaces_0167.png" alt="输入图像: image_static_stylized_dining_spaces_0167" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同一输入图像 + 同一 prompt · 分差 0.8980</div>
      <h4><code>image_static_stylized_dining_spaces_0167</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan right. Dining spaces with umbrella stand, coat rack, and door. Camera pull out. A living room with sofa, coffee table, and television. Camera move right. A living room with countertop, stove, and refrigerator.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_dining_spaces_0167">
      <source src="../../../assets/metric_examples/2a35a862ded7__echo-infinity-image-static__image_static_stylized_dining_spaces_0167.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9995</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_dining_spaces_0167">
      <source src="../../../assets/metric_examples/3bcc10d6420d__hy_worldplay_ar_image_test_balanced_5000_5s__image_static_stylized_dining_spaces_0167.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.1015</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/c391d632722b__image_static_photorealistic_suburb_0038.jpg" alt="输入图像: image_static_photorealistic_suburb_0038" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同一输入图像 + 同一 prompt · 分差 0.8980</div>
      <h4><code>image_static_photorealistic_suburb_0038</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move right. A park path with trees, playground, and city skyline. Camera pull out. A park path with park entrance, trees, and parking lot. Camera pan right. A suburban street with houses, parked cars, and city skyline.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_suburb_0038">
      <source src="../../../assets/metric_examples/b7739b09ef44__hy_worldplay_ar_image_test_balanced_5000_5s__image_static_photorealistic_suburb_0038.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_suburb_0038">
      <source src="../../../assets/metric_examples/6b9218e3501f__cosmos3-nano-image-static-8gpu-singleload-20260612_170108__image_static_photorealistic_suburb_0038.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.1020</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/625825740365__image_static_photorealistic_workspaces_0169.jpeg" alt="输入图像: image_static_photorealistic_workspaces_0169" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同一输入图像 + 同一 prompt · 分差 0.8979</div>
      <h4><code>image_static_photorealistic_workspaces_0169</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera orbit right. A modern office lounge with gray sofa, wall art, and seating.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_workspaces_0169">
      <source src="../../../assets/metric_examples/77151f030349__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_workspaces_0169.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_workspaces_0169">
      <source src="../../../assets/metric_examples/00568367d116__cosmos3-nano-image-static-8gpu-singleload-20260612_170108__image_static_photorealistic_workspaces_0169.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.1021</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
