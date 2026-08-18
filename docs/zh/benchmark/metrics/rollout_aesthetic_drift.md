# Rollout Aesthetic Drift

**rollout_aesthetic_drift** 衡量 rollout 内的**美学时序退化**。逐帧 LAION 美学分数划分为 $N=10$ 个等长片段；漂移为最佳与最差片段均值之间的相对下降。

\[
D_{\mathrm{aes}} = \frac{\bar{s}_{\mathrm{best}} - \bar{s}_{\mathrm{worst}}}{|\bar{s}_{\mathrm{best}}|}
\tag{1}
\]

越低越好，结果钳位至 $[0,1]$。仅基于预测。后端：`quality_drift`（美学分量经 pyiqa `laion_aes`）。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/08a7ae54eb87__image_static_stylized_living_spaces_0080__input.jpg" alt="输入图像: image_static_stylized_living_spaces_0080" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4898</div>
      <h4><code>image_static_stylized_living_spaces_0080</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pull out. A living room corner with bookshelf, potted plant, and yellow sectional sofa. Camera move left. A living room corner with framed photos, console table, and hallway. Camera pan left. A dining area with dining table, chairs, and wooden paneling.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_living_spaces_0080">
          <source src="../../../assets/metric_examples/7b9bb137b4d6__sana-wm-image-static__image_static_stylized_living_spaces_0080.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_living_spaces_0080">
          <source src="../../../assets/metric_examples/f87734cf7816__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_living_spaces_0080.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.5102</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/27ef0171b751__image_static_photorealistic_terrestrial_landscape_0100__input.jpg" alt="输入图像: image_static_photorealistic_terrestrial_landscape_0100" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.5320</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0100</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A mountain landscape with valley, river, and mountain peaks. Camera pull out. A mountain landscape with forest, wildlife, and deer. Camera move right. A mountain landscape with village, smoke, and chimneys.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_terrestrial_landscape_0100">
          <source src="../../../assets/metric_examples/3ef2096fbf6e__stable_virtual_camera__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9854</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_terrestrial_landscape_0100">
          <source src="../../../assets/metric_examples/1c7dbcf5ff29__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.4535</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/2cd9a570f49f__image_static_stylized_living_spaces_0139__input.jpg" alt="输入图像: image_static_stylized_living_spaces_0139" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4699</div>
      <h4><code>image_static_stylized_living_spaces_0139</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A japanese living room with tatami mats, tea set, and low table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_living_spaces_0139">
          <source src="../../../assets/metric_examples/2ef034707832__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_living_spaces_0139.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9885</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_living_spaces_0139">
          <source src="../../../assets/metric_examples/835e8a94f624__sana-wm-image-static__image_static_stylized_living_spaces_0139.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.5186</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_aesthetic_drift</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "LAION-Aesthetics Predictor (CLIP-based linear aesthetic regressor)"

    Used via pyiqa `laion_aes`; CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings.
