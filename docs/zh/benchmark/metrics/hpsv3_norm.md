# HPSv3-Norm

HPSv3-Norm 采用 HPSv3 奖励模型度量生成视频的**人类偏好对齐视觉质量**。该指标**仅需 prediction**：各评测帧独立打分，无需与 reference rollout 配对。

`hpsv3_norm` 是正式的 `image_static` quality 指标，同时进入 `per_metric_summary` 与七项等权 quality 总分。

WorldArena 从 prediction 视频中均匀子采样至多 20 帧，默认以 HPSv3 官方评测文档中提供的批大小（`batch_size: 8`）推理；这也是 8×A100 视频评测中实测的吞吐折中点。批处理只改变执行方式：每帧仍独立获得一个奖励；为保证可比性，同一次横向评测应使用相同的批大小。对帧 $x_i$，HPSv3 推理器输出原始奖励 $r_i$。每视频平均原始奖励为

\[
\bar{r}
=
\frac{1}{N}\sum_{i=1}^{N} r_i
\tag{1}
\]

设固定归一化分位数为 $p_1$、$p_{99}$（默认 $p_1 = 5.21$，$p_{99} = 8.66$）。$[0, 100]$ 上的归一化得分为

\[
S_{\mathrm{HPS}}
=
\mathrm{clip}\!\left(
\frac{\bar{r} - p_1}{p_{99} - p_1} \times 100,\,
0,\,
100
\right)
\tag{2}
\]

WorldArena 将指标 raw 值存为 $[0, 1]$ 区间内的 $S_{\mathrm{HPS}} / 100$。得分越高表示与人类偏好对齐程度越强。若 HPSv3 backend 或 checkpoint 不可用，指标记为 `not_applicable`。

## 参考文献 {#reference}

!!! quote "HPSv3 — Human Preference Score"

    基于人类偏好监督的视觉-语言奖励模型；WorldArena 对每视频平均原始奖励施加线性分位数归一化。

<!-- metric-examples:start -->
## 人工精选的同 Case 对比

下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d6b1f731ad67__image_static_photorealistic_aquatic_landscape_0001.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0001" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">人工复核 · 同一 case、输入与 prompt · gap 0.1702</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>实际生成 prompt</span>
        <p>Camera orbit left. A coastal landscape with ocean waves, distant hills, and sunset sky.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">较高分结果保持了连贯且细节丰富的日落景观；较低分结果更平、更缺乏偏好一致性。HPSv3 只评价生成帧本身。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_aquatic_landscape_0001">
          <source src="../../../assets/metric_examples/e3f37ff52cf2__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.1702</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>hpsv3_norm</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_aquatic_landscape_0001">
          <source src="../../../assets/metric_examples/curated_20260801/9d372e4f2066__image_static_photorealistic_aquatic_landscape_0001__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>hpsv3_norm</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
