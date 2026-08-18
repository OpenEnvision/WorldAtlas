# Epipolar Geometry

本页定义三组 **GVGC-style 极线几何诊断指标**，用于检验生成视频相邻帧是否具备稳定的双视角极线解释 [[Hartley and Zisserman, 2004](#reference)]。它们只依赖 prediction 视频本身，不需要相机位姿标注或 reference video；核心问题是：局部特征对应关系是否近似满足 epipolar constraint，从而说明两帧能否被同一个 pinhole 相机模型下的相对运动解释。

在经典多视角几何中，若两帧来自同一场景、仅发生刚体相机运动，则第一帧中的点 \(x\) 在第二帧中的对应点 \(x'\) 必然落在由基础矩阵 \(F\) 诱导的极线 \(l' = F x\) 上。生成视频若出现背景扭曲、物体局部漂移、深度层次不稳定，或相邻帧并非同一三维结构的投影，RANSAC inlier 将难以同时满足该约束，Sampson 误差会升高、inlier 比例下降。因此，极线指标提供一种**轻量、可解释**的双视角几何信号；但它只检查稀疏特征点的 epipolar 一致性，不能替代 MEt3R、重投影误差或重建类指标对全局三维自洽性的判断。

WorldArena 在 prediction 的采样帧上，按 `pair_stride` 构造相邻（或跨步）帧对，默认使用 SIFT 特征与 Lowe 比率测试进行匹配，再以 OpenCV RANSAC 估计基础矩阵 \(F\)。对 RANSAC inlier 计算 Sampson 误差，并统计 inlier 比例与覆盖率；若主路径失败，runtime 还可依次尝试放宽匹配阈值或 Shi-Tomasi + Lucas-Kanade 跟踪 fallback。三个子指标共享同一次 backend 分析，分别报告如下：

| 指标 | 原始统计量 | 方向 |
| --- | --- | --- |
| `epipolar_error` | 基础矩阵 RANSAC 后 inlier 的 Sampson 误差中位数 | 越低越好 |
| `epipolar_inlier_rate` | RANSAC inlier 数 / Lowe 比率过滤后的匹配数 | 越高越好 |
| `feature_match_coverage` | RANSAC inlier 数 / 较少一侧关键点数 | 越高越好 |

对匹配齐次点 \(x_i, x'_i\)，Sampson 误差为

\[
d_i = \frac{|{x'_i}^{\mathsf T} F x_i|}{\sqrt{(Fx_i)_1^2 + (Fx_i)_2^2 + (F^{\mathsf T}x'_i)_1^2 + (F^{\mathsf T}x'_i)_2^2}}
\tag{1}
\]

每个有效帧对以 RANSAC inlier 的 \(d_i\) 中位数作为 pair 级 `epipolar_error`；视频级取各 pair 中位数，并在 \([0, 5]\) 上反向归一化。

每个有效帧对的 inlier 率与覆盖率为

\[
r = \frac{\#\mathrm{inliers}}{\#\mathrm{good\ matches}},
\qquad
c = \min\!\left(\frac{\#\mathrm{inliers}}{\min(\#\mathrm{kp}_\mathrm{left}, \#\mathrm{kp}_\mathrm{right})},\, 1\right)
\tag{2}
\]

视频级 `epipolar_inlier_rate` 与 `feature_match_coverage` 分别对 \(r\)、\(c\) 取均值，在 \([0,1]\) 上恒等归一化。`epipolar_error` 取各有效帧对中位数的中位数，并在 \([0, 5]\) 上反向归一化到 \([0, 1]\)（归一化分数越高表示极线一致性越好）。若所有帧对都无法形成足够 inlier，三项均返回 `not_applicable`。这些指标当前为 **diagnostic**，单独报告，不进 headline 聚合；在弱纹理、强非刚性运动或 dynamic prompt 场景下可能频繁失效，宜与 MEt3R、重投影误差等互补阅读。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/ec72668a50dc__image_static_photorealistic_public_spaces_0064__input.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0064" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.0743</div>
      <h4><code>image_static_photorealistic_public_spaces_0064</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move left. A prison cell with barred door, cobwebs, and peeling paint.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_public_spaces_0064">
          <source src="../../../assets/metric_examples/curated_20260801/394f94fcdcda__image_static_photorealistic_public_spaces_0064__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9924</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>epipolar_error</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_public_spaces_0064">
          <source src="../../../assets/metric_examples/curated_20260801/448f0f8062ae__image_static_photorealistic_public_spaces_0064__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.9181</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>epipolar_error</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/a7645e0fbde8__image_static_photorealistic_living_spaces_0155__input.jpg" alt="输入图像: image_static_photorealistic_living_spaces_0155" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.0770</div>
      <h4><code>image_static_photorealistic_living_spaces_0155</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A vintage bathroom with wooden shelves, bath products, and towels.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_living_spaces_0155">
          <source src="../../../assets/metric_examples/curated_20260801/adf26cec582b__image_static_photorealistic_living_spaces_0155__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9911</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>epipolar_error</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_living_spaces_0155">
          <source src="../../../assets/metric_examples/curated_20260801/c28e8366ed77__image_static_photorealistic_living_spaces_0155__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.9141</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>epipolar_error</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/bdb608aac636__image_static_stylized_living_spaces_0037__input.jpg" alt="输入图像: image_static_stylized_living_spaces_0037" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.0860</div>
      <h4><code>image_static_stylized_living_spaces_0037</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A bedroom corner with wardrobe, chair, and rug.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_living_spaces_0037">
          <source src="../../../assets/metric_examples/curated_20260801/7f6f8eb8d3f9__image_static_stylized_living_spaces_0037__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9871</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>epipolar_error</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_living_spaces_0037">
          <source src="../../../assets/metric_examples/c223556eb328__matrix_game2_5s_image_static__image_static_stylized_living_spaces_0037.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.9011</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>epipolar_error</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c298f05f69b4__image_static_stylized_public_spaces_0049__input.jpg" alt="输入图像: image_static_stylized_public_spaces_0049" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.7193</div>
      <h4><code>image_static_stylized_public_spaces_0049</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit left. A library aisle with bookshelves, books, and reading table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_public_spaces_0049">
          <source src="../../../assets/metric_examples/curated_20260801/080234cb97e9__image_static_stylized_public_spaces_0049__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9841</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_public_spaces_0049">
          <source src="../../../assets/metric_examples/curated_20260801/d609d31a9446__image_static_stylized_public_spaces_0049__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.2648</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/e1cb08434f22__image_static_photorealistic_public_spaces_0154__input.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0154" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.6446</div>
      <h4><code>image_static_photorealistic_public_spaces_0154</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit left. A library aisle with bookshelves, books, and reading tables.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_public_spaces_0154">
          <source src="../../../assets/metric_examples/curated_20260801/645f013eaa1d__image_static_photorealistic_public_spaces_0154__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9924</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_public_spaces_0154">
          <source src="../../../assets/metric_examples/curated_20260801/238c47aefdbf__image_static_photorealistic_public_spaces_0154__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3478</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/a896a0aef634__image_static_stylized_city_0157__input.jpg" alt="输入图像: image_static_stylized_city_0157" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.6917</div>
      <h4><code>image_static_stylized_city_0157</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit right. A voxel city street with brick buildings, crosswalk, and street signs.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_city_0157">
          <source src="../../../assets/metric_examples/curated_20260801/0c0c6f4e64c1__image_static_stylized_city_0157__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9927</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_city_0157">
          <source src="../../../assets/metric_examples/curated_20260801/2844ad2d0729__image_static_stylized_city_0157__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3010</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/14a72672e0d0__image_static_stylized_living_spaces_0134__input.jpg" alt="输入图像: image_static_stylized_living_spaces_0134" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8146</div>
      <h4><code>image_static_stylized_living_spaces_0134</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit right. A modern living room with bookshelf, abstract painting, and sectional sofa.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_living_spaces_0134">
          <source src="../../../assets/metric_examples/curated_20260801/356f3599300f__image_static_stylized_living_spaces_0134__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.8414</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_living_spaces_0134">
          <source src="../../../assets/metric_examples/curated_20260801/e28be2531442__image_static_stylized_living_spaces_0134__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0268</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>feature_match_coverage</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/7ee246743d64__image_static_stylized_living_spaces_0091__input.jpg" alt="输入图像: image_static_stylized_living_spaces_0091" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8123</div>
      <h4><code>image_static_stylized_living_spaces_0091</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A bedroom corner with wardrobe, small bookshelf, and window.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_living_spaces_0091">
          <source src="../../../assets/metric_examples/curated_20260801/7e539eca98ce__image_static_stylized_living_spaces_0091__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.8396</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_living_spaces_0091">
          <source src="../../../assets/metric_examples/curated_20260801/1752059b5701__image_static_stylized_living_spaces_0091__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0273</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>feature_match_coverage</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/624e8a1fef57__image_static_photorealistic_workspaces_0083__input.jpg" alt="输入图像: image_static_photorealistic_workspaces_0083" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8311</div>
      <h4><code>image_static_photorealistic_workspaces_0083</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. An elegant office with bookshelf, books, and armchair.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_workspaces_0083">
          <source src="../../../assets/metric_examples/curated_20260801/ce09026b33ba__image_static_photorealistic_workspaces_0083__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.8699</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_workspaces_0083">
          <source src="../../../assets/metric_examples/curated_20260801/e5b5f4c35b8e__image_static_photorealistic_workspaces_0083__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0388</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>feature_match_coverage</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "Multiple View Geometry in Computer Vision"

    Hartley & Zisserman, Cambridge University Press, 2004.
