# Rollout Imaging Drift

**rollout_imaging_drift** 衡量 rollout 内的**成像质量时序退化**。逐帧 MUSIQ 分数划分为 $N=10$ 个等长片段；漂移为最佳与最差片段均值之间的相对下降。

\[
D_{\mathrm{img}} = \frac{\bar{s}_{\mathrm{best}} - \bar{s}_{\mathrm{worst}}}{|\bar{s}_{\mathrm{best}}|}
\tag{1}
\]

越低越好，结果钳位至 $[0,1]$。仅基于预测。后端：`quality_drift`（成像分量经 MUSIQ-SPAQ）。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/a547760fd431__image_static_photorealistic_aquatic_landscape_0071__input.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0071" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.7798</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0071</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A castle on rocky cliff with rocky path and sea. Camera pan right. A castle on rocky cliff with castle entrance, drawbridge, and forest. Camera pull out. A forest edge with castle, twilight sky, and distant sea.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_aquatic_landscape_0071">
          <source src="../../../assets/metric_examples/468c4e319b29__yume_sample5b__image_static_photorealistic_aquatic_landscape_0071.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9751</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_imaging_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_aquatic_landscape_0071">
          <source src="../../../assets/metric_examples/66ef97296842__warp-as-history-image-static-8gpu-20260612_152000__image_static_photorealistic_aquatic_landscape_0071.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1953</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_imaging_drift</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/41926cb7db86__image_static_stylized_passageways_0014__input.jpg" alt="输入图像: image_static_stylized_passageways_0014" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8269</div>
      <h4><code>image_static_stylized_passageways_0014</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit right. A passageways scene with staircase, column, and arched passageway.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_passageways_0014">
          <source src="../../../assets/metric_examples/0c627e2bd9b7__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_stylized_passageways_0014.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9804</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_imaging_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_passageways_0014">
          <source src="../../../assets/metric_examples/dacc3200ce74__matrix_game2_5s__image_static_stylized_passageways_0014.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1534</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_imaging_drift</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/3a6a439006d3__image_static_photorealistic_passageways_0081__input.jpg" alt="输入图像: image_static_photorealistic_passageways_0081" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.7934</div>
      <h4><code>image_static_photorealistic_passageways_0081</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A long corridor with corridor extension, columns, and mural.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_passageways_0081">
          <source src="../../../assets/metric_examples/4300ac0093a0__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_passageways_0081.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9573</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_imaging_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_passageways_0081">
          <source src="../../../assets/metric_examples/519973dc5747__matrix_game2_5s__image_static_photorealistic_passageways_0081.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1639</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_imaging_drift</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "MUSIQ: Multi-scale Image Quality Transformer"

    Zejiang Hou, Hui Ren, Ming Sun, Jiahao Li, Weining Wang.  
    ICCV 2021.  
    [https://arxiv.org/abs/2108.05997](https://arxiv.org/abs/2108.05997)
