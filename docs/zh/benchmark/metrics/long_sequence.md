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
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和完全一致的展示 prompt，然后比较模型生成结果。paired 展示必须满足 input image 相同、prompt 文本相同，并且一个模型高分、一个模型低分。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `long_sequence_aesthetic_quality`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/b3d0e7407370__image_static_stylized_terrestrial_landscape_0004.jpg" alt="输入图像: image_static_stylized_terrestrial_landscape_0004" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 0.3237</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0004</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move right. A snowy forest with snow-covered trees, distant path, and snowy ground.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_terrestrial_landscape_0004">
      <source src="../../../assets/metric_examples/95a9cc54e18b__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_terrestrial_landscape_0004.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.6700</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_terrestrial_landscape_0004">
      <source src="../../../assets/metric_examples/cf80f8e9ac0c__motionctrl__image_static_stylized_terrestrial_landscape_0004.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3464</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f26edd43f07f__image_static_stylized_passageways_0105.png" alt="输入图像: image_static_stylized_passageways_0105" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同一输入图像 + 同一 prompt · 分差 0.2892</div>
      <h4><code>image_static_stylized_passageways_0105</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera push in. A stylized indoor passageway with wooden floor, bookshelves, and reading area.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_passageways_0105">
      <source src="../../../assets/metric_examples/f8f86fde3531__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_passageways_0105.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.5670</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_passageways_0105">
      <source src="../../../assets/metric_examples/58ddbf8228dd__motionctrl__image_static_stylized_passageways_0105.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.2777</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2c2b9c480799__image_static_photorealistic_terrestrial_landscape_0100.jpg" alt="输入图像: image_static_photorealistic_terrestrial_landscape_0100" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同一输入图像 + 同一 prompt · 分差 0.2665</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0100</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan right. A mountain landscape with valley, river, and mountain peaks. Camera pull out. A mountain landscape with forest, wildlife, and deer. Camera move right. A mountain landscape with village, smoke, and chimneys.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_terrestrial_landscape_0100">
      <source src="../../../assets/metric_examples/3ef2096fbf6e__stable_virtual_camera__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.6438</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_terrestrial_landscape_0100">
      <source src="../../../assets/metric_examples/b61a742a411b__motionctrl__image_static_photorealistic_terrestrial_landscape_0100.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3773</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/1a7dd6bacd65__image_static_photorealistic_suburb_0063.png" alt="输入图像: image_static_photorealistic_suburb_0063" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同一输入图像 + 同一 prompt · 分差 0.2628</div>
      <h4><code>image_static_photorealistic_suburb_0063</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan right. A suburban street with houses, parked cars, and street signs.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_suburb_0063">
      <source src="../../../assets/metric_examples/2ad22bc01a86__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_suburb_0063.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.5978</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_suburb_0063">
      <source src="../../../assets/metric_examples/ebf0aaa3ac2b__motionctrl__image_static_photorealistic_suburb_0063.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3350</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/810078323779__image_static_photorealistic_verdant_landscape_0080.jpg" alt="输入图像: image_static_photorealistic_verdant_landscape_0080" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同一输入图像 + 同一 prompt · 分差 0.2586</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0080</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera orbit left. An overgrown greenhouse with vines, rusted tools, and workbenches.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_verdant_landscape_0080">
      <source src="../../../assets/metric_examples/1f27af206163__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_verdant_landscape_0080.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.6437</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_verdant_landscape_0080">
      <source src="../../../assets/metric_examples/b8d768559d11__motionctrl__image_static_photorealistic_verdant_landscape_0080.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3851</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/a68afb03ba7f__image_static_stylized_public_spaces_0115.png" alt="输入图像: image_static_stylized_public_spaces_0115" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同一输入图像 + 同一 prompt · 分差 0.2425</div>
      <h4><code>image_static_stylized_public_spaces_0115</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan left. A vintage museum with display cases, historical artifacts, and inventions.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_public_spaces_0115">
      <source src="../../../assets/metric_examples/963841dad8d8__stable_virtual_camera__image_static_stylized_public_spaces_0115.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.6808</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_public_spaces_0115">
      <source src="../../../assets/metric_examples/dfa37f7aa924__motionctrl__image_static_stylized_public_spaces_0115.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.4384</div>
  </div>
</figure>
  </div>
</article>
</div>
### `long_sequence_imaging_quality`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/13486b394bf9__image_static_photorealistic_verdant_landscape_0109.jpeg" alt="输入图像: image_static_photorealistic_verdant_landscape_0109" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 0.4753</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0109</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan right. A forest edge with wildflowers and distant hills.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_verdant_landscape_0109">
      <source src="../../../assets/metric_examples/73755855521a__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_verdant_landscape_0109.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.8024</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_verdant_landscape_0109">
      <source src="../../../assets/metric_examples/4ac9274ee922__motionctrl__image_static_photorealistic_verdant_landscape_0109.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3270</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/8fcb15010df2__image_static_photorealistic_public_spaces_0068.png" alt="输入图像: image_static_photorealistic_public_spaces_0068" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同一输入图像 + 同一 prompt · 分差 0.4353</div>
      <h4><code>image_static_photorealistic_public_spaces_0068</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera push in. An abandoned chapel with altar, carvings, and faded symbols.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_public_spaces_0068">
      <source src="../../../assets/metric_examples/5009426ec26e__yume-sample5b-image-static-dynamic-5s__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.7406</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_public_spaces_0068">
      <source src="../../../assets/metric_examples/a14a53e19198__warp-as-history-image-static-8gpu-20260612_152000__image_static_photorealistic_public_spaces_0068.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3053</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/4460d14379f4__image_static_stylized_suburb_0084.jpg" alt="输入图像: image_static_stylized_suburb_0084" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同一输入图像 + 同一 prompt · 分差 0.4182</div>
      <h4><code>image_static_stylized_suburb_0084</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pull out. A suburban intersection with trees, parked cars, and pedestrians.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_suburb_0084">
      <source src="../../../assets/metric_examples/0d348dbc85f1__motionctrl__image_static_stylized_suburb_0084.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.7578</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_suburb_0084">
      <source src="../../../assets/metric_examples/ac7c105f46d1__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_suburb_0084.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3397</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/72dc3ca43c46__image_static_stylized_verdant_landscape_0151.png" alt="输入图像: image_static_stylized_verdant_landscape_0151" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同一输入图像 + 同一 prompt · 分差 0.3910</div>
      <h4><code>image_static_stylized_verdant_landscape_0151</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move right. A rural landscape with path, tall grasses, and traditional buildings.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_verdant_landscape_0151">
      <source src="../../../assets/metric_examples/34239ebf5935__yume-sample5b-missing-reverse__image_static_stylized_verdant_landscape_0151.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.7832</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_verdant_landscape_0151">
      <source src="../../../assets/metric_examples/13ccb050b001__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_verdant_landscape_0151.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3922</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/266a02cd8d70__image_static_stylized_aquatic_landscape_0053.jpg" alt="输入图像: image_static_stylized_aquatic_landscape_0053" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同一输入图像 + 同一 prompt · 分差 0.3801</div>
      <h4><code>image_static_stylized_aquatic_landscape_0053</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera move left. A rocky shoreline with vegetation, distant hills, and waves.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_aquatic_landscape_0053">
      <source src="../../../assets/metric_examples/cdd233ddb905__motionctrl__image_static_stylized_aquatic_landscape_0053.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.7774</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_aquatic_landscape_0053">
      <source src="../../../assets/metric_examples/51b05aeee0a9__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_aquatic_landscape_0053.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3974</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/03873b9fe77e__image_static_stylized_terrestrial_landscape_0149.png" alt="输入图像: image_static_stylized_terrestrial_landscape_0149" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同一输入图像 + 同一 prompt · 分差 0.3767</div>
      <h4><code>image_static_stylized_terrestrial_landscape_0149</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera pan left. A desert landscape with dry shrubs, distant mesas, and rocky terrain. Camera move left. A desert landscape with distant mesas, rocky terrain, and dry shrubs. Camera pull out. A desert landscape with open sky, distant mesas, and rocky terrain.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_stylized_terrestrial_landscape_0149">
      <source src="../../../assets/metric_examples/476b02851a40__yume-sample5b-missing-reverse-5s-target__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.7517</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_stylized_terrestrial_landscape_0149">
      <source src="../../../assets/metric_examples/30b3f591dec6__matrix_game2_5s__image_static_stylized_terrestrial_landscape_0149.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.3750</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
