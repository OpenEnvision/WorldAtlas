# Long Sequence Aesthetic Quality

**long_sequence_aesthetic_quality** 衡量预测帧的**无参考美学吸引力均值**。每帧 $I_t$ 经冻结 CLIP ViT-L/14 编码，LAION-Aesthetics 线性头输出归一化至 $[0,1]$ 的逐帧分数。

\[
S_{\mathrm{aes}} = \frac{1}{T}\sum_{t=1}^{T} s_t^{\mathrm{aes}}
\tag{1}
\]

越高越好。该指标衡量长 rollout 质量，仅基于预测，由 `long_sequence` 后端（`aesthetic_quality` 维度）计算。共享运行时见 [Long-Sequence 指标](long_sequence.md)。

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
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "LAION-Aesthetics Predictor (CLIP-based linear aesthetic regressor)"

    Used via pyiqa `laion_aes`; CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings.
