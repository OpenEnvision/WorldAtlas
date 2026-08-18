# Motion Smoothness

Motion Smoothness 在帧插值先验下衡量生成视频的**时序可插值性**。VFIMamba 由相邻偶数帧重建被丢弃的奇数帧；重建误差越低表示运动越平滑连续，而非语义或幅度正确。

对窗口 \(k\)，\(\hat{I}_{2k+1} = \Phi(I_{2k}, I_{2k+2})\) 与真实奇数帧 \(I_{2k+1}\) 通过 MSE、SSIM、LPIPS 比较。原始分量对 \(K\) 个窗口取均值：

\[
E_{\mathrm{mse}} = \frac{1}{K}\sum_k \mathrm{MSE}(\hat{I}_{2k+1}, I_{2k+1}),\quad
S_{\mathrm{ssim}} = \frac{1}{K}\sum_k \mathrm{SSIM}(\hat{I}_{2k+1}, I_{2k+1}),\quad
E_{\mathrm{lpips}} = \frac{1}{K}\sum_k \mathrm{LPIPS}(\hat{I}_{2k+1}, I_{2k+1})
\tag{1}
\]

计算 MSE 前会先把解码得到的 `uint8` 帧转换为 `float32`，避免负像素差在减法和平方时按模 256 回绕。

各分量独立归一化后取算术均值。\(S_{\mathrm{smooth}}\) 越高表示生成运动越平滑。计算成功时为 **official**，无需参考视频。

当前配置中的 MSE 经验边界早于 `v2_float32_mse`。修复后的原始报告会带版本标记，但在用修正后的 benchmark 全量样本重新标定该边界之前，其归一化分数仍应视为临时值。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/928a4bb3a461__image_dynamic_photorealistic_deformable_0096__input.jpg" alt="输入图像: image_dynamic_photorealistic_deformable_0096" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8951</div>
      <h4><code>image_dynamic_photorealistic_deformable_0096</code></h4>
      <div class="metric-case-meta"><span>套件: image_dynamic</span><span>场景: photorealistic / deformable</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera fixed. Heavy wheat grain heads nod downward and lift slightly as wind moves through the dense cluster. Overlapping stalks brush together while the warm field backdrop stays consistent.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_dynamic_photorealistic_deformable_0096">
          <source src="../../../assets/metric_examples/curated_20260801/154e282065d6__image_dynamic_photorealistic_deformable_0096__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9462</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>motion_smoothness</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_dynamic_photorealistic_deformable_0096">
          <source src="../../../assets/metric_examples/curated_20260801/120a8a389b9d__image_dynamic_photorealistic_deformable_0096__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0511</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>motion_smoothness</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/b2b909fd3a6c__image_dynamic_photorealistic_articulated_0136__input.jpg" alt="输入图像: image_dynamic_photorealistic_articulated_0136" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.9357</div>
      <h4><code>image_dynamic_photorealistic_articulated_0136</code></h4>
      <div class="metric-case-meta"><span>套件: image_dynamic</span><span>场景: photorealistic / articulated</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera fixed. A cheetah sprints across the dry grassy savannah, its legs cycling rapidly and body stretching low with each stride. Golden grass, sunlight, and shadows remain consistent around it.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_dynamic_photorealistic_articulated_0136">
          <source src="../../../assets/metric_examples/b03e9f8572a3__matrix-game-3-image-dynamic-5s__image_dynamic_photorealistic_articulated_0136.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9557</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>motion_smoothness</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_dynamic_photorealistic_articulated_0136">
          <source src="../../../assets/metric_examples/curated_20260801/4a46c83ac1c0__image_dynamic_photorealistic_articulated_0136__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0200</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>motion_smoothness</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/934774d4baa7__image_dynamic_stylized_deformable_0154__input.jpg" alt="输入图像: image_dynamic_stylized_deformable_0154" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.8808</div>
      <h4><code>image_dynamic_stylized_deformable_0154</code></h4>
      <div class="metric-case-meta"><span>套件: image_dynamic</span><span>场景: stylized / deformable</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera fixed. A vibrant stylized wave curls and rolls over the rocky shoreline, its colorful water swirling forward. Clouds twist overhead while the rocks and grass patch remain stable.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_dynamic_stylized_deformable_0154">
          <source src="../../../assets/metric_examples/curated_20260801/84fe2b2a1fd7__image_dynamic_stylized_deformable_0154__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9577</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>motion_smoothness</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_dynamic_stylized_deformable_0154">
          <source src="../../../assets/metric_examples/curated_20260801/ff7cca037561__image_dynamic_stylized_deformable_0154__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0769</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>motion_smoothness</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "VFIMamba: Video Frame Interpolation with State Space Models"

    Zhang et al., NeurIPS 2024.

!!! quote "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Zhang et al., CVPR 2018.
