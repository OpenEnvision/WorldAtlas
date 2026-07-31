# Temporal Calibration

Temporal Calibration 度量生成视频是否维持**稳定的物理时钟**：当文件声明某个播放帧率时，画面中的运动是否对应相同的真实时间流逝？WorldArena 在 `real_time` 维度中以 output-only audit 形式调用被包装的 Pulse-of-Motion / Visual Chronometer 预测器 [[Gao et al., 2026](#reference-visual-chronometer)]。

这与 optical flow、motion magnitude 或 temporal smoothness 不同。一段 rollout 可以逐帧非常平滑，却在时间上仍然错误：坠落可能像被放慢的重力，快速周期性运动也可能显得不自然地迟缓。Visual Chronometer 将这类失败称为 **chronometric hallucination**——视觉连贯，但物理时间尺度模糊、不稳定或不可控。

预测器从重叠时间窗口估计 **Physical FPS**（`PhyFPS`），而不是单独相信容器元数据。WorldArena 将平均推断帧率与 nominal playback FPS $f_{\mathrm{meta}}$（OpenCV 读取的 container FPS，或 manifest 回退值）比较，并报告三个正式指标：`physical_fps_error`、`physical_fps_pct_error` 与 `physical_fps_intra_video_cv`。前两项刻画全局节奏偏差，第三项刻画单条视频内部的节奏漂移。

对每个 prediction，runtime 以默认参数 `clip_length = 30`、`stride = 4`、`resolution = 216` 提取重叠窗口，每个窗口得到一个片段估计 $\hat{f}_k$。平均推断物理 FPS 为

\[
\bar{f}_{\mathrm{phy}}
=
\frac{1}{K}\sum_{k=1}^{K} \hat{f}_k
\tag{1}
\]

绝对与相对标定误差分别为

\[
E_{\mathrm{abs}} = \left|\bar{f}_{\mathrm{phy}} - f_{\mathrm{meta}}\right|
\tag{2}
\]

\[
E_{\mathrm{pct}} = \frac{E_{\mathrm{abs}}}{f_{\mathrm{meta}}}
\tag{3}
\]

视频内部不稳定性由片段预测的变异系数表示：

\[
C_{\mathrm{intra}}
=
\frac{\operatorname{std}(\hat{f}_1,\ldots,\hat{f}_K)}{\operatorname{mean}(\hat{f}_1,\ldots,\hat{f}_K)}
\tag{4}
\]

WorldArena 将 `physical_fps_error`、`physical_fps_pct_error`、`physical_fps_intra_video_cv` 分别反向缩放到 $[0,1]$（`physical_fps_error` 使用 $[0,30]$，其余两项使用 $[0,1]$）。**归一化分数越高，表示时间标定越好。** 若 side runtime、checkpoint 或 FPS 元数据不可用，整组指标记为 `not_applicable`。

## 参考文献 {#reference-visual-chronometer}

!!! quote "The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"

    Xiangbo Gao, Mingyang Wu, Siyuan Yang, Jiongze Yu, Pardis Taghavi, Fangzhou Lin, and Zhengzhong Tu.  
    arXiv:2603.14375, 2026.

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/2eb4a88b4229__image_static_stylized_suburb_0123__input.jpg" alt="输入图像: image_static_stylized_suburb_0123" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.9833</div>
      <h4><code>image_static_stylized_suburb_0123</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move left. A historic courtyard with garden, pathways, and seating.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条结果展示的是同一庭院。请重点观察播放节奏：较高分结果的估计物理 FPS 与容器 FPS 几乎一致；较低分结果的运动速度则远快于其声明帧率。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_suburb_0123">
          <source src="../../../assets/metric_examples/curated_20260801/7baeb4773bfa__image_static_stylized_suburb_0123__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9933</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>physical_fps_error</span><span>估计物理 FPS 23.8 · 声明 FPS 24.0 · 绝对误差 0.2</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_suburb_0123">
          <source src="../../../assets/metric_examples/curated_20260801/8fca43aff466__image_static_stylized_suburb_0123__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0100</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>physical_fps_error</span><span>估计物理 FPS 44.7 · 声明 FPS 15.0 · 绝对误差 29.7</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/0f505edb11c8__image_static_photorealistic_workspaces_0144__input.jpg" alt="输入图像: image_static_photorealistic_workspaces_0144" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.7192</div>
      <h4><code>image_static_photorealistic_workspaces_0144</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera move left. A modern conference room with side wall, whiteboard, and chairs. Camera pull out. A modern conference room with entrance door, coat rack, and chairs. Camera pan left. A modern conference room with hallway and artwork.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分的会议室镜头保持近乎恒定的物理 FPS 节奏；较低分结果从开头的快速运动骤降为后段的慢速运动。该指标不评价画面外观质量。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_workspaces_0144">
          <source src="../../../assets/metric_examples/curated_20260801/61c84b8ed7cc__image_static_photorealistic_workspaces_0144__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9894</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>physical_fps_intra_video_cv</span><span>段间物理 FPS 69.1–71.7 · 变异系数 0.011</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_workspaces_0144">
          <source src="../../../assets/metric_examples/curated_20260801/ed03c3e72ce1__image_static_photorealistic_workspaces_0144__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.2702</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>physical_fps_intra_video_cv</span><span>段间物理 FPS 11.9–54.8 · 变异系数 0.730</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/547f0dde32ed__image_static_photorealistic_dining_spaces_0137__input.jpg" alt="输入图像: image_static_photorealistic_dining_spaces_0137" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.9983</div>
      <h4><code>image_static_photorealistic_dining_spaces_0137</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit left. A cozy dining area with bar counter, stools, and tables.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分环绕镜头的估计物理 FPS 与声明 FPS 对齐；较低分结果约以 60 物理 FPS 运动，却只声明 13 FPS，因此相对误差很大。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_dining_spaces_0137">
          <source src="../../../assets/metric_examples/curated_20260801/10332b72a711__image_static_photorealistic_dining_spaces_0137__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9983</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>physical_fps_pct_error</span><span>估计物理 FPS 60.1 · 声明 FPS 60.0 · 相对误差 0.17%</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_dining_spaces_0137">
          <source src="../../../assets/metric_examples/curated_20260801/41740b9f9a00__image_static_photorealistic_dining_spaces_0137__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>physical_fps_pct_error</span><span>估计物理 FPS 59.8 · 声明 FPS 13.0 · 相对误差 360.00%</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
