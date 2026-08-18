# 片段连续性（Segment Continuity）

片段连续性度量生成视频是否出现**检测到的场景切换**。非预期的硬切会破坏时序连贯性，反映片段不连续。

TransNetV2 为解码帧 $x_i$ 输出场景边界概率 $p_i$。当 $p_i > \theta$ 时标记为切点候选（默认 $\theta = 0.5$）。切点记录遵循 TransNetV2 的 0→1 跃迁解析：在帧索引 $i$ 处由 $0$ 升至 $1$ 即记为一次切换。

为抑制误检，切点候选需满足最小场景长度 $L$（默认 $L = 10$ 帧）：任一保留切点两侧的片段长度均不得短于 $L$。设单视频保留切点数为 $n_{\mathrm{cut}}$，则 per-video 得分为二值：

\[
S_{\mathrm{seg}}
=
\begin{cases}
1, & n_{\mathrm{cut}} = 0 \\
0, & n_{\mathrm{cut}} > 0
\end{cases}
\tag{1}
\]

对 $N$ 个视频，模型级聚合为

\[
\bar{S}_{\mathrm{seg}}
=
1 - \frac{n_{\mathrm{cut}}}{N}
\tag{2}
\]

其中 $n_{\mathrm{cut}}$ 统计至少含一个保留切点的视频数。WorldArena 将式 (1) 的二值结果存为 per-sample raw 分数。该指标**仅需 prediction**，且依赖 TransNetV2 推理代码与权重。

## 参考文献 {#reference}

!!! quote "TransNet V2 — Shot Boundary Detection"

    Sočik et al., ACM MM 2020.
    [https://arxiv.org/abs/2008.04838](https://arxiv.org/abs/2008.04838)

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/6419698d992c__image_static_photorealistic_city_0064__input.jpg" alt="输入图像: image_static_photorealistic_city_0064" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 1.0000</div>
      <h4><code>image_static_photorealistic_city_0064</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pull out. An expanded city view with buildings, distant hills, and extended waterfront.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_city_0064">
          <source src="../../../assets/metric_examples/curated_20260801/4133494bf815__image_static_photorealistic_city_0064__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>segment_continuity</span><span>检测到的硬切次数: 0</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_city_0064">
          <source src="../../../assets/metric_examples/curated_20260801/0dffa172da47__image_static_photorealistic_city_0064__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>segment_continuity</span><span>检测到的硬切次数: 1</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/051f91d93080__image_static_photorealistic_aquatic_landscape_0084__input.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0084" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 1.0000</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0084</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move left. A lake edge with reeds, autumn trees, and distant mountains.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_aquatic_landscape_0084">
          <source src="../../../assets/metric_examples/curated_20260801/3ff6bbb0893e__image_static_photorealistic_aquatic_landscape_0084__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>segment_continuity</span><span>检测到的硬切次数: 0</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_aquatic_landscape_0084">
          <source src="../../../assets/metric_examples/curated_20260801/fe8753fa0234__image_static_photorealistic_aquatic_landscape_0084__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>segment_continuity</span><span>检测到的硬切次数: 1</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/6e723fd86c7d__image_static_photorealistic_aquatic_landscape_0089__input.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0089" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 1.0000</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0089</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pull out. A rocky shoreline with trees and distant hills. Camera move right. A forest edge with dense foliage and sunlight. Camera pan right. A forest edge with forest path, scattered rocks, and tree canopy.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_aquatic_landscape_0089">
          <source src="../../../assets/metric_examples/curated_20260801/78d59f9021b3__image_static_photorealistic_aquatic_landscape_0089__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>segment_continuity</span><span>检测到的硬切次数: 0</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_aquatic_landscape_0089">
          <source src="../../../assets/metric_examples/curated_20260801/0b291d32d947__image_static_photorealistic_aquatic_landscape_0089__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>segment_continuity</span><span>检测到的硬切次数: 1</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
