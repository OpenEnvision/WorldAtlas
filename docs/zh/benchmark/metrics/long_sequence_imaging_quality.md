# Long Sequence Imaging Quality

**long_sequence_imaging_quality** 使用 MUSIQ 衡量预测帧的**无参考技术成像质量均值**（清晰度、噪声、压缩伪影）。native 分数除以 $100$ 后取帧均值，映射至 $[0,1]$。

\[
S_{\mathrm{img}} = \frac{1}{T}\sum_{t=1}^{T} s_t^{\mathrm{img}}
\tag{1}
\]

越高越好。该指标衡量长 rollout 质量，仅基于预测，由 `long_sequence` 后端（`imaging_quality` 维度；默认 MUSIQ-SPAQ 权重，长边 resize 至 $512$）计算。共享运行时见 [Long-Sequence 指标](long_sequence.md)。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/297e8735b260__image_static_photorealistic_public_spaces_0068__input.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0068" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4353</div>
      <h4><code>image_static_photorealistic_public_spaces_0068</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera push in. An abandoned chapel with altar, carvings, and faded symbols.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_public_spaces_0068">
          <source src="../../../assets/metric_examples/5009426ec26e__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7406</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_public_spaces_0068">
          <source src="../../../assets/metric_examples/a14a53e19198__warp-as-history-image-static-8gpu-20260612_152000__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3053</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/8ff65b592716__image_static_stylized_terrestrial_landscape_0149__input.jpg" alt="输入图像: image_static_stylized_terrestrial_landscape_0149" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.3767</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0149</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan left. A desert landscape with dry shrubs, distant mesas, and rocky terrain. Camera move left. A desert landscape with distant mesas, rocky terrain, and dry shrubs. Camera pull out. A desert landscape with open sky, distant mesas, and rocky terrain.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_terrestrial_landscape_0149">
          <source src="../../../assets/metric_examples/476b02851a40__yume-sample5b-missing-reverse-5s-target__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7517</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>long_sequence_imaging_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_terrestrial_landscape_0149">
          <source src="../../../assets/metric_examples/30b3f591dec6__matrix_game2_5s__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3750</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>long_sequence_imaging_quality</span></figcaption>
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
