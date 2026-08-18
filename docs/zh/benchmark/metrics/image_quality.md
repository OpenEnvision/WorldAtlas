# Image Quality（MUSIQ）

`image_quality` 是正式的无参考 `image_static` quality 指标。它使用 MUSIQ-SPAQ 图像质量模型评估 prediction 帧，不需要 paired reference rollout。

对协议选出的每个 prediction 帧 $I_i$，MUSIQ 输出原生 $[0,100]$ 分数 $q_i$。WorldArena 先逐帧归一化并截断：

\[
s_i = \mathrm{clip}\!\left(\frac{q_i}{100}, 0, 1\right),
\]

再对归一化帧分数取平均：

\[
S_{\mathrm{MUSIQ}} = \frac{1}{N}\sum_{i=1}^{N}s_i.
\]

得分越高越好。该结果进入 `per_metric_summary` 与七项等权 quality 总分。默认 backend 为 `musiq`，预处理模式为 `longer`。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/4369893ac62b__image_static_stylized_living_spaces_0096__input.jpg" alt="输入图像: image_static_stylized_living_spaces_0096" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4837</div>
      <h4><code>image_static_stylized_living_spaces_0096</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A modern living space with hallway, neon lights, and artwork.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_living_spaces_0096">
          <source src="../../../assets/metric_examples/curated_20260801/1e7a3127e127__image_static_stylized_living_spaces_0096__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7218</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_living_spaces_0096">
          <source src="../../../assets/metric_examples/curated_20260801/b8886f3dbd11__image_static_stylized_living_spaces_0096__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.2381</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/871a90dfcd05__image_static_photorealistic_passageways_0129__input.jpg" alt="输入图像: image_static_photorealistic_passageways_0129" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4037</div>
      <h4><code>image_static_photorealistic_passageways_0129</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A rustic hallway with wooden floor, window with curtains, and vintage furniture.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_passageways_0129">
          <source src="../../../assets/metric_examples/curated_20260801/6ae6bcb192f7__image_static_photorealistic_passageways_0129__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7462</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_passageways_0129">
          <source src="../../../assets/metric_examples/curated_20260801/72ff72478035__image_static_photorealistic_passageways_0129__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3425</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/554f035b9d9c__image_static_photorealistic_passageways_0140__input.jpg" alt="输入图像: image_static_photorealistic_passageways_0140" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4016</div>
      <h4><code>image_static_photorealistic_passageways_0140</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A luxurious hallway with marble columns, doors, and gold-framed mirrors.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_passageways_0140">
          <source src="../../../assets/metric_examples/curated_20260801/8cdcc534d30e__image_static_photorealistic_passageways_0140__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7334</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_passageways_0140">
          <source src="../../../assets/metric_examples/curated_20260801/5b233f91cddf__image_static_photorealistic_passageways_0140__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3318</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
