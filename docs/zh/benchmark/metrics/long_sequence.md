# Long-Sequence 指标

WorldArena 内置了 `Long-Sequence` backend，用来从完整 prediction video 中提取长时运动和画质信号。
这组指标并非仅考察单帧或相邻两帧，而是将生成视频划分为多个短 clip，并进一步分析更长 horizon 上的运动连续性和视觉质量。

它主要回答的问题是：生成视频在更长 horizon 上是否还能保持运动连续性和画质。

这组指标是 prediction-only 指标，不使用 benchmark reference rollout，也不需要标注。
因此它们更适合解释长视频退化模式，例如运动停滞、突兀时间伪影、画质随时间下降等。

## 覆盖的指标

| 指标名 | 包装后的 Long-Sequence 维度 | 在 WorldArena 中的角色 |
| --- | --- | --- |
| `long_sequence_motion_smoothness` | `motion_smoothness` | 基于视频插帧误差的运动平滑度 |
| `long_sequence_dynamic_degree` | `dynamic_degree` | 基于光流强度的运动是否充分发生 |
| `long_sequence_aesthetic_quality` | `aesthetic_quality` | 基于无参考美学模型的视觉审美质量 |
| `long_sequence_imaging_quality` | `imaging_quality` | 基于 MUSIQ 的成像清晰度和退化质量 |

## 原理说明

普通短程指标容易漏掉只在更长 rollout 后出现的问题。
Long-Sequence backend 会先把完整视频切成多个短 clip，然后把原生 long-video 例程用于运动和画质维度。

`motion_smoothness` 用 AMT-S 做中间帧插值，比较插值结果和真实中间帧的差异。
`dynamic_degree` 用 RAFT 估计光流强度，判断视频是否有足够运动。
`aesthetic_quality` 和 `imaging_quality` 分别使用质量模型对每个 clip 的帧进行无参考打分。

## 具体算法

1. WorldArena 读取完整 prediction video，并在临时 workspace 中准备 Long-Sequence 输入目录。
2. `worldarena.benchmark.long_video` 会把长视频切成 `split_clip/`，并生成每个维度需要的最小 manifest。
3. dispatcher 根据请求维度调用对应实现：
    - `motion_smoothness`：AMT-S 插帧后比较预测中间帧和真实中间帧。
    - `dynamic_degree`：RAFT 估计光流，并统计高运动 transition 是否达到阈值。
    - `aesthetic_quality`：无参考 aesthetic scorer 对 clip 帧打分。
    - `imaging_quality`：MUSIQ 对 clip 帧打分。
4. 每个 WorldArena 指标只读取自己对应的 wrapped dimension。
5. wrapper 会记录运行时元数据，包括实际 device、source root、clip 数量、per-video scores 和维度特定细节。
6. 最后每个指标按 `config/benchmark.yaml` 中的 `[0,1]` 范围独立归一化；这四个指标都是越高越好。

## 关键公式

对 `motion_smoothness`，AMT-S 会用偶数采样帧插值得到中间帧，并和真实中间帧比较平均绝对差：

!!! info "运动平滑度"

    \[
    s_{\mathrm{smooth}}
    =
    \operatorname{clamp}
    \left(
        \frac{255-\operatorname{mean}(|I^{real}_{mid}-I^{interp}_{mid}|)}{255},
        0,
        1
    \right)
    \]

对 `dynamic_degree`，RAFT 会估计相邻采样帧之间的光流，并取光流幅值最高的 5% 像素均值作为 transition motion score。
当超过运动阈值的 transition 数量达到计数阈值时，该 clip 记为有动态，输出 `1.0`；否则输出 `0.0`。

对 `aesthetic_quality` 和 `imaging_quality`，backend 分别对每个 clip 的帧计算无参考质量分数，再对 clip、video 和整批样本求平均。

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/f2f1a5bacaf1__image_static_stylized_terrestrial_landscape_0001__input.jpg" alt="输入图像: image_static_stylized_terrestrial_landscape_0001" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4214</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A stone church with pathway, wildflowers, and distant hills.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_stylized_terrestrial_landscape_0001">
          <source src="../../../assets/metric_examples/5a5135ed4314__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_terrestrial_landscape_0001.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.8615</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_visual_score</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_stylized_terrestrial_landscape_0001">
          <source src="../../../assets/metric_examples/c643ca835a4a__stable_virtual_camera__image_static_stylized_terrestrial_landscape_0001.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.4401</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_visual_score</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/8eea82dddee2__image_static_photorealistic_suburb_0067__input.jpg" alt="输入图像: image_static_photorealistic_suburb_0067" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.4277</div>
      <h4><code>image_static_photorealistic_suburb_0067</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera push in. A suburban street with house, front yard, and garden.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_suburb_0067">
          <source src="../../../assets/metric_examples/41f4e3499573__ac3d-image-static__image_static_photorealistic_suburb_0067.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.8223</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_visual_score</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_suburb_0067">
          <source src="../../../assets/metric_examples/65b2958f9c9a__warp-as-history-image-static-8gpu-20260612_152000__image_static_photorealistic_suburb_0067.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.3946</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_visual_score</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/712d1288bb7b__image_static_photorealistic_city_0080__input.jpg" alt="输入图像: image_static_photorealistic_city_0080" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.3887</div>
      <h4><code>image_static_photorealistic_city_0080</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit left. An abandoned parking lot with weeds, derelict buildings, and side street.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_city_0080">
          <source src="../../../assets/metric_examples/cb357d2cde00__wan22-image-static-8gpu-20260612_160808__image_static_photorealistic_city_0080.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.8288</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>rollout_visual_score</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_city_0080">
          <source src="../../../assets/metric_examples/da3e80c4c375__sana-wm-image-static__image_static_photorealistic_city_0080.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.4400</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>rollout_visual_score</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
