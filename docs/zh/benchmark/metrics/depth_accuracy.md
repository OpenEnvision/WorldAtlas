# Depth Accuracy

Depth Accuracy 是配对指标，比较预测与参考锚帧的单目相对深度结构。两侧经同一 Depth Anything V2 模型推理，不使用 ground-truth 深度。

对第 \(i\) 对，设 \(D_i^{\mathrm{pred}}\)、\(D_i^{\mathrm{ref}}\) 为深度图。中位数比率对齐：

\[
\alpha_i = \frac{\mathrm{median}(D_i^{\mathrm{ref}})}{\max(\mathrm{median}(D_i^{\mathrm{pred}}), 10^{-6})},
\qquad
\tilde{D}_i^{\mathrm{pred}} = \alpha_i D_i^{\mathrm{pred}}
\tag{1}
\]

在有效像素 \(\Omega_i = \{(x,y) \mid D_i^{\mathrm{ref}}(x,y) > 10^{-3}\}\) 上，帧级绝对相对误差为

\[
e_i = \frac{1}{|\Omega_i|}\sum_{(x,y)\in\Omega_i} \frac{\left|\tilde{D}_i^{\mathrm{pred}}(x,y) - D_i^{\mathrm{ref}}(x,y)\right|}{\max(D_i^{\mathrm{ref}}(x,y), 10^{-6})}
\tag{2}
\]

原始得分 \(e_{\mathrm{depth}} = \frac{1}{N}\sum_i e_i\) 对 \(N\) 组对齐帧对取均值。误差越低，归一化得分越高。该指标评估粗粒度前后景排序是否保留，而非 metric 深度真值或相机轨迹精度。

<!-- metric-examples:start -->
## 代表性同 Case 对比

每个展示块先给出该 case 的 input image 和实际生成 prompt，然后比较模型生成结果。paired 展示必须通过 sample_id、输入图像内容 SHA-256 和实际生成 prompt SHA-256 三重一致性校验。示例按写实/风格化、室内/室外、场景类别及分差层级进行均衡选择，而不是只挑最高分差。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。

机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.

### `depth_accuracy`

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-header">
    <div class="metric-case-kicker">Case 1 · 当前快照极值样例</div>
  <p class="metric-case-note">当前正式快照中该 metric 没有跨模型同 case 配对；以下为各自独立的高分/低分极值样例，而非同一输入下的模型对比。</p>
  </div>
  <div class="metric-fallback-stack">
  <section class="metric-fallback-side metric-fallback-side--good">
    <div class="metric-fallback-side-heading">高分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/e480336c5c00__image_static_stylized_dining_spaces_0071.jpg" alt="输入图像: image_static_stylized_dining_spaces_0071" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_stylized_dining_spaces_0071</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera move left. A modern kitchen with dining table, chairs, and wooden floor. Camera pan left. A modern kitchen with window, potted plant, and curtains. Camera pull out. A modern kitchen with kitchen island, bar stools, and hanging lights.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: ac3d-image-static image_static_stylized_dining_spaces_0071">
      <source src="../../../assets/metric_examples/afbfb9bd5380__ac3d-image-static__image_static_stylized_dining_spaces_0071.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9934</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>好结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  <section class="metric-fallback-side metric-fallback-side--bad">
    <div class="metric-fallback-side-heading">低分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/20f6a24367ac__image_static_photorealistic_aquatic_landscape_0017.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0017" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_aquatic_landscape_0017</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera orbit left. An aquatic landscape with red lighthouse, ship&#x27;s deck, and harbor.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: ac3d-image-static image_static_photorealistic_aquatic_landscape_0017">
      <source src="../../../assets/metric_examples/d339e4209040__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0017.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>差结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-header">
    <div class="metric-case-kicker">Case 2 · 当前快照极值样例</div>
  <p class="metric-case-note">当前正式快照中该 metric 没有跨模型同 case 配对；以下为各自独立的高分/低分极值样例，而非同一输入下的模型对比。</p>
  </div>
  <div class="metric-fallback-stack">
  <section class="metric-fallback-side metric-fallback-side--good">
    <div class="metric-fallback-side-heading">高分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/a4762377f6eb__image_static_photorealistic_public_spaces_0138.jpg" alt="输入图像: image_static_photorealistic_public_spaces_0138" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_public_spaces_0138</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera orbit right. A grand room with ornate chandelier, detailed ceiling, and arched windows.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: ac3d-image-static image_static_photorealistic_public_spaces_0138">
      <source src="../../../assets/metric_examples/9be3503b5aa7__ac3d-image-static__image_static_photorealistic_public_spaces_0138.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9925</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>好结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  <section class="metric-fallback-side metric-fallback-side--bad">
    <div class="metric-fallback-side-heading">低分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/07428555b67e__image_static_photorealistic_aquatic_landscape_0129.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0129" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_aquatic_landscape_0129</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera pan left. An aquatic landscape with beach, umbrellas, and people.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: ac3d-image-static image_static_photorealistic_aquatic_landscape_0129">
      <source src="../../../assets/metric_examples/de45932a63fc__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0129.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>差结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-header">
    <div class="metric-case-kicker">Case 3 · 当前快照极值样例</div>
  <p class="metric-case-note">当前正式快照中该 metric 没有跨模型同 case 配对；以下为各自独立的高分/低分极值样例，而非同一输入下的模型对比。</p>
  </div>
  <div class="metric-fallback-stack">
  <section class="metric-fallback-side metric-fallback-side--good">
    <div class="metric-fallback-side-heading">高分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/ee734879694d__image_static_photorealistic_dining_spaces_0149.png" alt="输入图像: image_static_photorealistic_dining_spaces_0149" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_dining_spaces_0149</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / dining_spaces</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera move right. A modern kitchen with dining table, chairs, and wooden floor.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: ac3d-image-static image_static_photorealistic_dining_spaces_0149">
      <source src="../../../assets/metric_examples/fc545d0de2d7__ac3d-image-static__image_static_photorealistic_dining_spaces_0149.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9921</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>好结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  <section class="metric-fallback-side metric-fallback-side--bad">
    <div class="metric-fallback-side-heading">低分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/05bb7d71ea29__image_static_photorealistic_city_0168.jpg" alt="输入图像: image_static_photorealistic_city_0168" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_city_0168</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / city</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera orbit right. A city street with stadium, towers, and olympic rings.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: ac3d-image-static image_static_photorealistic_city_0168">
      <source src="../../../assets/metric_examples/9df30d500edc__ac3d-image-static__image_static_photorealistic_city_0168.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>差结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--fallback">
  <div class="metric-case-header">
    <div class="metric-case-kicker">Case 4 · 当前快照极值样例</div>
  <p class="metric-case-note">当前正式快照中该 metric 没有跨模型同 case 配对；以下为各自独立的高分/低分极值样例，而非同一输入下的模型对比。</p>
  </div>
  <div class="metric-fallback-stack">
  <section class="metric-fallback-side metric-fallback-side--good">
    <div class="metric-fallback-side-heading">高分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/aafb975766fe__image_static_photorealistic_dining_spaces_0071.jpg" alt="输入图像: image_static_photorealistic_dining_spaces_0071" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_dining_spaces_0071</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / dining_spaces</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera push in. A modern kitchen with dining table, breakfast items, and brown chairs.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: ac3d-image-static image_static_photorealistic_dining_spaces_0071">
      <source src="../../../assets/metric_examples/a36f7bd6076d__ac3d-image-static__image_static_photorealistic_dining_spaces_0071.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9920</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>好结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  <section class="metric-fallback-side metric-fallback-side--bad">
    <div class="metric-fallback-side-heading">低分极值样例</div>
    <div class="metric-case-context">
      <figure class="metric-input-card">
        <img src="../../../assets/metric_examples/inputs/bd04feec3cfa__image_static_photorealistic_terrestrial_landscape_0007.jpg" alt="输入图像: image_static_photorealistic_terrestrial_landscape_0007" loading="lazy" decoding="async">
        <figcaption>输入图像</figcaption>
      </figure>
      <div class="metric-case-copy">
        <h4><code>image_static_photorealistic_terrestrial_landscape_0007</code></h4>
        <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
        <div class="metric-case-prompt">
          <span>目标 prompt</span>
          <p>Camera pan left. A mountain landscape with road, trucks, and clouds. Camera move left. A mountain landscape with road, village, and houses. Camera pull out. A mountain landscape with road, plains, and clouds.</p>
          <small>prompt 来源: prompt_target</small>
        </div>
      </div>
    </div>
    <div class="metric-output-grid metric-output-grid--single">
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: ac3d-image-static image_static_photorealistic_terrestrial_landscape_0007">
      <source src="../../../assets/metric_examples/a513b129439f__ac3d-image-static__image_static_photorealistic_terrestrial_landscape_0007.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>差结果</strong>
    <span>指标: depth_accuracy · 协议: segment_random</span>
  </figcaption>
</figure>
    </div>
  </section>
  </div>
</article>
</div>

<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "Depth Anything V2"

    Yang et al., 2024.
    [https://arxiv.org/abs/2406.09414](https://arxiv.org/abs/2406.09414)
