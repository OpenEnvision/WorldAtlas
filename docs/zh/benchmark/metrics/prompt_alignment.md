# Prompt Alignment

Prompt Alignment 衡量生成视频在 benchmark 协议锚点帧上与目标指令 \(T = \texttt{sample.prompt\_target}\) 的**语义一致程度**。它评估预测在采样时刻是否实现所请求的场景内容，而非整段 rollout 的时序连贯性或视觉真实性。

记 \(\mathcal{A} = \{f_i\}_{i=1}^{M}\) 为评测协议从解码预测中选取的锚点帧集合。设 CLIP ViT-B/16 图像编码器为 \(\phi_I\)、文本编码器为 \(\phi_T\)，L2 归一化嵌入为 \(\mathbf{e}_i = \phi_I(f_i)\)、\(\mathbf{t} = \phi_T(T)\)。帧 \(i\) 的原生 CLIPScore 为

\[
s_i = \max\!\left(100 \cdot \langle \mathbf{e}_i, \mathbf{t} \rangle,\, 0\right)
\tag{1}
\]

WorldArena 将各锚点得分映射至 \([0,1]\)：\(\tilde{s}_i = \mathrm{clamp}(s_i / 100,\, 0,\, 1)\)，并报告均值

\[
S_{\mathrm{prompt}} = \frac{1}{M}\sum_{i=1}^{M} \tilde{s}_i
\tag{2}
\]

\(S_{\mathrm{prompt}}\) 越高，表示协议锚点处的指令跟随越强。当存在显式指令监督且 conditioning 非自参照（`conditioning_strategy ≠ self`）时，该指标为 **official**。图像套件可以单帧图像为条件，但模型输出仍须为视频。

下方案例从完整的 2026-07-17 `image_static` 快照中人工筛选：20 个模型在相同的 3,372 个 case 上评测。每组左右视频都经过严格校验，sample ID、条件图像 SHA-256 和实际生成 prompt SHA-256 完全一致，且两条 generation record 都明确把该图作为 conditioning input；“好/差”仅描述 Prompt Alignment，不代表整体视频质量。每个分数均为四个 `segment_random` 协议锚点的平均值。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/acb20f90f503__image_static_stylized_public_spaces_0029__input.jpg" alt="输入图像: image_static_stylized_public_spaces_0029" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.1526</div>
      <h4><code>image_static_stylized_public_spaces_0029</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit left. A vintage car display with wooden floor, framed artworks, and cars.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_public_spaces_0029">
          <source src="../../../assets/metric_examples/curated_20260801/cc11f324a93f__image_static_stylized_public_spaces_0029__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.3508</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>prompt_alignment</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_public_spaces_0029">
          <source src="../../../assets/metric_examples/curated_20260801/117df6803bf2__image_static_stylized_public_spaces_0029__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1982</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>prompt_alignment</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/066d09fa43d6__image_static_stylized_workspaces_0016__input.jpg" alt="输入图像: image_static_stylized_workspaces_0016" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.1561</div>
      <h4><code>image_static_stylized_workspaces_0016</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A cozy workspace with desk, plants, and window.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_workspaces_0016">
          <source src="../../../assets/metric_examples/curated_20260801/68b11dff54be__image_static_stylized_workspaces_0016__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.3342</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>prompt_alignment</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_workspaces_0016">
          <source src="../../../assets/metric_examples/cea803b80e21__matrix_game2_5s_image_static__image_static_stylized_workspaces_0016.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1781</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>prompt_alignment</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/bd9b5b37aa13__image_static_stylized_workspaces_0044__input.jpg" alt="输入图像: image_static_stylized_workspaces_0044" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.1533</div>
      <h4><code>image_static_stylized_workspaces_0044</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pan right. A modern loft bed with bookshelf, indoor plant, and wall.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_workspaces_0044">
          <source src="../../../assets/metric_examples/curated_20260801/3e816f47f052__image_static_stylized_workspaces_0044__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.3331</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>prompt_alignment</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_workspaces_0044">
          <source src="../../../assets/metric_examples/curated_20260801/153b900ccf6f__image_static_stylized_workspaces_0044__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1798</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>prompt_alignment</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
