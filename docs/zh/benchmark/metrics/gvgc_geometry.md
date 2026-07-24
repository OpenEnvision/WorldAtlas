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
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和实际生成 prompt，然后比较模型生成结果。paired 展示必须通过 sample_id、输入图像内容 SHA-256 和实际生成 prompt SHA-256 三重一致性校验。示例按写实/风格化、室内/室外、场景类别及分差层级进行均衡选择，而不是只挑最高分差。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `epipolar_error`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2792761326fc__image_static_photorealistic_workspaces_0016.jpg" alt="输入图像: image_static_photorealistic_workspaces_0016" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同 case 严格配对 · 分差 0.0678</div>
      <h4><code>image_static_photorealistic_workspaces_0016</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan right. A modern office with desks, computers, and office chairs. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_workspaces_0016">
      <source src="../../../assets/metric_examples/5186200d0d8d__astra_image_static__image_static_photorealistic_workspaces_0016.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9898</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_workspaces_0016">
      <source src="../../../assets/metric_examples/101a8b600780__matrix_game2_5s_image_static__image_static_photorealistic_workspaces_0016.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9221</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/51ab9c2d87a1__image_static_photorealistic_public_spaces_0014.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0014" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同 case 严格配对 · 分差 0.0537</div>
      <h4><code>image_static_photorealistic_public_spaces_0014</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan right. A pet care room with pet cages, feeding station, and water bowls.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_public_spaces_0014">
      <source src="../../../assets/metric_examples/e0104d4a4ae9__luciddreamer-image-static-8gpu-20260613_151844__image_static_photorealistic_public_spaces_0014.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9865</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_public_spaces_0014">
      <source src="../../../assets/metric_examples/b0fc8fda531d__ac3d-image-static__image_static_photorealistic_public_spaces_0014.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9329</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/20a32d64c96b__image_static_photorealistic_terrestrial_landscape_0091.jpg" alt="输入图像: image_static_photorealistic_terrestrial_landscape_0091" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同 case 严格配对 · 分差 0.0827</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0091</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pull out. An edge of the field with grasses and distant trees. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_terrestrial_landscape_0091">
      <source src="../../../assets/metric_examples/7992ecd2bebe__astra_image_static__image_static_photorealistic_terrestrial_landscape_0091.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_terrestrial_landscape_0091">
      <source src="../../../assets/metric_examples/4b8207797696__dreamx_world_image_static__image_static_photorealistic_terrestrial_landscape_0091.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9173</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/5b4d1e8308f0__image_static_photorealistic_aquatic_landscape_0070.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0070" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同 case 严格配对 · 分差 0.0545</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0070</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan left. A frozen lake with snow-covered mountains, evergreen trees, and distant peak. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_aquatic_landscape_0070">
      <source src="../../../assets/metric_examples/95c7578a9527__hy_worldplay_ar_image_test_balanced_5000_5s__image_static_photorealistic_aquatic_landscape_0070.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9875</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_aquatic_landscape_0070">
      <source src="../../../assets/metric_examples/b226b2177cb9__motionctrl_image_static_official_default_tmp2__image_static_photorealistic_aquatic_landscape_0070.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9330</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/bdb608aac636__image_static_stylized_living_spaces_0037.jpg" alt="输入图像: image_static_stylized_living_spaces_0037" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同 case 严格配对 · 分差 0.0704</div>
      <h4><code>image_static_stylized_living_spaces_0037</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan right. A bedroom corner with wardrobe, chair, and rug. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_living_spaces_0037">
      <source src="../../../assets/metric_examples/96bc5f185eee__dreamx_world_image_static__image_static_stylized_living_spaces_0037.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9715</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_living_spaces_0037">
      <source src="../../../assets/metric_examples/c223556eb328__matrix_game2_5s_image_static__image_static_stylized_living_spaces_0037.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9011</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/300dd624b49c__image_static_stylized_dining_spaces_0097.jpg" alt="输入图像: image_static_stylized_dining_spaces_0097" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同 case 严格配对 · 分差 0.0532</div>
      <h4><code>image_static_stylized_dining_spaces_0097</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pull out. Dining spaces with wooden table, chairs, and window with curtains. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_dining_spaces_0097">
      <source src="../../../assets/metric_examples/254315e5aac8__cameractrl_image_static__image_static_stylized_dining_spaces_0097.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9890</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_dining_spaces_0097">
      <source src="../../../assets/metric_examples/7d93ea0bf915__dreamx_world_image_static__image_static_stylized_dining_spaces_0097.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9358</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/185f3ac27937__image_static_stylized_verdant_landscape_0087.png" alt="输入图像: image_static_stylized_verdant_landscape_0087" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 7 · 同 case 严格配对 · 分差 0.0705</div>
      <h4><code>image_static_stylized_verdant_landscape_0087</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan left. A forest at sunset with trees, open field, and forest edge.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_verdant_landscape_0087">
      <source src="../../../assets/metric_examples/d3048e41a43a__longcat-video-image-static-8gpu-20260612_133950__image_static_stylized_verdant_landscape_0087.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_verdant_landscape_0087">
      <source src="../../../assets/metric_examples/5ab7297b6b75__ac3d-image-static__image_static_stylized_verdant_landscape_0087.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9295</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/85c22133f3df__image_static_stylized_city_0051.jpg" alt="输入图像: image_static_stylized_city_0051" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 8 · 同 case 严格配对 · 分差 0.0537</div>
      <h4><code>image_static_stylized_city_0051</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan right. A city construction site with street, snow, and cars.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_city_0051">
      <source src="../../../assets/metric_examples/d614d31c9a9c__longcat-video-image-static-8gpu-20260612_133950__image_static_stylized_city_0051.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9920</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_city_0051">
      <source src="../../../assets/metric_examples/2ec012332022__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_city_0051.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.9383</div>
  </div>
</figure>
  </div>
</article>
</div>
### `epipolar_inlier_rate`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/82eb4ea72231__image_static_photorealistic_passageways_0079.jpg" alt="输入图像: image_static_photorealistic_passageways_0079" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同 case 严格配对 · 分差 0.5194</div>
      <h4><code>image_static_photorealistic_passageways_0079</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move right. A bright hallway with wooden floor, armchair, and small table. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_passageways_0079">
      <source src="../../../assets/metric_examples/2aca94b55772__cameractrl_image_static__image_static_photorealistic_passageways_0079.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9344</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_passageways_0079">
      <source src="../../../assets/metric_examples/67bd53a27f54__matrix_game2_5s_image_static__image_static_photorealistic_passageways_0079.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.4150</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d8f00f994fc2__image_static_photorealistic_public_spaces_0150.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0150" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同 case 严格配对 · 分差 0.2254</div>
      <h4><code>image_static_photorealistic_public_spaces_0150</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan left. A historic building with flower beds, stone path, and hedges. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_public_spaces_0150">
      <source src="../../../assets/metric_examples/df2b86a89f74__dreamx_world_image_static__image_static_photorealistic_public_spaces_0150.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9759</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_public_spaces_0150">
      <source src="../../../assets/metric_examples/8d82cdd290f9__matrix_game2_5s_image_static__image_static_photorealistic_public_spaces_0150.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.7505</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/212961fe4813__image_static_photorealistic_verdant_landscape_0091.jpg" alt="输入图像: image_static_photorealistic_verdant_landscape_0091" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同 case 严格配对 · 分差 0.5209</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0091</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move left. Tall grasses with field of grasses, distant trees, and open sky. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_verdant_landscape_0091">
      <source src="../../../assets/metric_examples/4b1f1aa1624b__matrix_game2_5s_image_static__image_static_photorealistic_verdant_landscape_0091.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9878</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_verdant_landscape_0091">
      <source src="../../../assets/metric_examples/fe05997e84a6__hy_worldplay_ar_image_test_balanced_5000_5s__image_static_photorealistic_verdant_landscape_0091.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.4669</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/6e5579841b03__image_static_photorealistic_terrestrial_landscape_0006.jpg" alt="输入图像: image_static_photorealistic_terrestrial_landscape_0006" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同 case 严格配对 · 分差 0.2005</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0006</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit left. An indoor pool with wooden railings, decorative plants, and upper level. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_terrestrial_landscape_0006">
      <source src="../../../assets/metric_examples/f1115249bb4c__hy_worldplay_ar_distilled_image_test_balanced_5000_5s__image_static_photorealistic_terrestrial_landscape_0006.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9675</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_terrestrial_landscape_0006">
      <source src="../../../assets/metric_examples/7dd67d4756c8__cameractrl_image_static__image_static_photorealistic_terrestrial_landscape_0006.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.7670</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/8085680bb8c5__image_static_stylized_workspaces_0008.jpg" alt="输入图像: image_static_stylized_workspaces_0008" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同 case 严格配对 · 分差 0.5392</div>
      <h4><code>image_static_stylized_workspaces_0008</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit left. A modern conference room with bookshelves, glass walls, and modern chairs. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_workspaces_0008">
      <source src="../../../assets/metric_examples/762d162b0f5f__hy_worldplay_ar_distilled_image_test_balanced_5000_5s__image_static_stylized_workspaces_0008.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9507</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_workspaces_0008">
      <source src="../../../assets/metric_examples/f724b74f4ea9__matrix_game3_5s_image_static_dynamic__image_static_stylized_workspaces_0008.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.4115</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/69e48ac4cce9__image_static_stylized_dining_spaces_0041.jpg" alt="输入图像: image_static_stylized_dining_spaces_0041" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同 case 严格配对 · 分差 0.2484</div>
      <h4><code>image_static_stylized_dining_spaces_0041</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pull out. A beachside path with sand, lanterns, and ocean. Camera pan left. A beachfront lounge with lounge chairs, palm trees, and ocean. Camera move left. A beach bar with bar counter, stools, and string lights.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_dining_spaces_0041">
      <source src="../../../assets/metric_examples/055610540354__luciddreamer-image-static-8gpu-20260613_151844__image_static_stylized_dining_spaces_0041.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9290</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_dining_spaces_0041">
      <source src="../../../assets/metric_examples/eb258637ccee__longcat-video-image-static-8gpu-20260612_133950__image_static_stylized_dining_spaces_0041.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.6806</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/09490cc3a111__image_static_stylized_city_0021.jpg" alt="输入图像: image_static_stylized_city_0021" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 7 · 同 case 严格配对 · 分差 0.4548</div>
      <h4><code>image_static_stylized_city_0021</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move right. A cityscape silhouette with towers, buildings, and bridge. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_city_0021">
      <source src="../../../assets/metric_examples/50a398808e7c__motionctrl_image_static_official_default_tmp2__image_static_stylized_city_0021.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9696</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_city_0021">
      <source src="../../../assets/metric_examples/26dc3b8b908f__astra_image_static__image_static_stylized_city_0021.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.5148</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/b0540bc5452a__image_static_stylized_aquatic_landscape_0123.png" alt="输入图像: image_static_stylized_aquatic_landscape_0123" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 8 · 同 case 严格配对 · 分差 0.2243</div>
      <h4><code>image_static_stylized_aquatic_landscape_0123</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan left. Dense trees with stone pathway and lake. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_aquatic_landscape_0123">
      <source src="../../../assets/metric_examples/089030954b5b__hy_worldplay_ar_distilled_image_test_balanced_5000_5s__image_static_stylized_aquatic_landscape_0123.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9741</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_aquatic_landscape_0123">
      <source src="../../../assets/metric_examples/684370a91577__motionctrl_image_static_official_default_tmp2__image_static_stylized_aquatic_landscape_0123.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.7498</div>
  </div>
</figure>
  </div>
</article>
</div>
### `feature_match_coverage`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/32d68cb087eb__image_static_photorealistic_passageways_0155.jpg" alt="输入图像: image_static_photorealistic_passageways_0155" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同 case 严格配对 · 分差 0.8407</div>
      <h4><code>image_static_photorealistic_passageways_0155</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera push in. A concrete corridor with end wall and light source. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_passageways_0155">
      <source src="../../../assets/metric_examples/35ab3fc3e20b__cameractrl_image_static__image_static_photorealistic_passageways_0155.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.8889</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_passageways_0155">
      <source src="../../../assets/metric_examples/bdf8a95737dc__matrix_game3_5s_image_static_dynamic__image_static_photorealistic_passageways_0155.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0482</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f5727e3d872e__image_static_photorealistic_workspaces_0074.png" alt="输入图像: image_static_photorealistic_workspaces_0074" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同 case 严格配对 · 分差 0.4021</div>
      <h4><code>image_static_photorealistic_workspaces_0074</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit left. A modern workspace with sneaker wall, vault door, and futuristic car. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_workspaces_0074">
      <source src="../../../assets/metric_examples/5627a1bd7949__hy_worldplay_ar_distilled_image_test_balanced_5000_5s__image_static_photorealistic_workspaces_0074.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.5724</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_workspaces_0074">
      <source src="../../../assets/metric_examples/ef518968e81d__hy_worldplay_ar_rl_image_test_balanced_5000_5s__image_static_photorealistic_workspaces_0074.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.1703</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/b553385fdcc6__image_static_photorealistic_verdant_landscape_0041.jpg" alt="输入图像: image_static_photorealistic_verdant_landscape_0041" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同 case 严格配对 · 分差 0.9874</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0041</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit right. A sunlit field with grass and distant trees. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_verdant_landscape_0041">
      <source src="../../../assets/metric_examples/ec70c09d7795__cameractrl_image_static__image_static_photorealistic_verdant_landscape_0041.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_verdant_landscape_0041">
      <source src="../../../assets/metric_examples/36fca0c55ef0__dreamx_world_image_static__image_static_photorealistic_verdant_landscape_0041.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0126</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/de6916bf7630__image_static_photorealistic_terrestrial_landscape_0067.jpg" alt="输入图像: image_static_photorealistic_terrestrial_landscape_0067" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同 case 严格配对 · 分差 0.3860</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0067</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pan left. A rural field with palm trees, haystacks, and green field. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_terrestrial_landscape_0067">
      <source src="../../../assets/metric_examples/01de9809e554__lyra-2-image-static-openenvision__image_static_photorealistic_terrestrial_landscape_0067.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.4469</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_terrestrial_landscape_0067">
      <source src="../../../assets/metric_examples/25022d74d777__matrix_game3_5s_image_static_dynamic__image_static_photorealistic_terrestrial_landscape_0067.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0609</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/569e5bf50a6a__image_static_stylized_living_spaces_0034.jpg" alt="输入图像: image_static_stylized_living_spaces_0034" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同 case 严格配对 · 分差 0.8906</div>
      <h4><code>image_static_stylized_living_spaces_0034</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move left. A minimalist living room with bookshelf, window, and armchair. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_living_spaces_0034">
      <source src="../../../assets/metric_examples/379d7e7af470__cameractrl_image_static__image_static_stylized_living_spaces_0034.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_living_spaces_0034">
      <source src="../../../assets/metric_examples/6f9f03923ec1__astra_image_static__image_static_stylized_living_spaces_0034.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.1094</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/033617ccf24c__image_static_stylized_public_spaces_0008.jpg" alt="输入图像: image_static_stylized_public_spaces_0008" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同 case 严格配对 · 分差 0.3882</div>
      <h4><code>image_static_stylized_public_spaces_0008</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit left. A gallery corridor with statues, paintings, and windows. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_public_spaces_0008">
      <source src="../../../assets/metric_examples/e8af08ddfa9e__astra_image_static__image_static_stylized_public_spaces_0008.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.4870</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_public_spaces_0008">
      <source src="../../../assets/metric_examples/2922278be5d3__matrix_game3_5s_image_static_dynamic__image_static_stylized_public_spaces_0008.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0988</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/bbdfe50ab941__image_static_stylized_aquatic_landscape_0048.jpg" alt="输入图像: image_static_stylized_aquatic_landscape_0048" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 7 · 同 case 严格配对 · 分差 0.9024</div>
      <h4><code>image_static_stylized_aquatic_landscape_0048</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit right. A sunset lake with rocks, sunset sky, and distant shoreline. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_aquatic_landscape_0048">
      <source src="../../../assets/metric_examples/f7d0b1135562__cameractrl_image_static__image_static_stylized_aquatic_landscape_0048.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_aquatic_landscape_0048">
      <source src="../../../assets/metric_examples/0920650a0969__matrix_game2_5s_image_static__image_static_stylized_aquatic_landscape_0048.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0976</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/3e8be99af0df__image_static_stylized_suburb_0007.jpg" alt="输入图像: image_static_stylized_suburb_0007" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 8 · 同 case 严格配对 · 分差 0.4055</div>
      <h4><code>image_static_stylized_suburb_0007</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera pull out. A path intersection with trees, benches, and people. Everything in the scene remains completely motionless.</p>
        <small>prompt 来源: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_suburb_0007">
      <source src="../../../assets/metric_examples/774124c6e96d__hy_worldplay_ar_image_test_balanced_5000_5s__image_static_stylized_suburb_0007.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.4402</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_suburb_0007">
      <source src="../../../assets/metric_examples/d5bd629c2cff__dreamx_world_image_static__image_static_stylized_suburb_0007.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0348</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "Multiple View Geometry in Computer Vision"

    Hartley & Zisserman, Cambridge University Press, 2004.
