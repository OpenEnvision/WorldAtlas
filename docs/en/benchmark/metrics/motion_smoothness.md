# Motion Smoothness

Motion Smoothness measures **temporal interpolatability** of a generated video under a frame-interpolation prior. VFIMamba reconstructs discarded odd frames from neighboring even frames; low reconstruction error indicates smooth, continuous motion—not correct semantics or magnitude.

For window \(k\), \(\hat{I}_{2k+1} = \Phi(I_{2k}, I_{2k+2})\) is compared to ground-truth odd frame \(I_{2k+1}\) via MSE, SSIM, and LPIPS. Raw components average over \(K\) windows:

\[
E_{\mathrm{mse}} = \frac{1}{K}\sum_k \mathrm{MSE}(\hat{I}_{2k+1}, I_{2k+1}),\quad
S_{\mathrm{ssim}} = \frac{1}{K}\sum_k \mathrm{SSIM}(\hat{I}_{2k+1}, I_{2k+1}),\quad
E_{\mathrm{lpips}} = \frac{1}{K}\sum_k \mathrm{LPIPS}(\hat{I}_{2k+1}, I_{2k+1})
\tag{1}
\]

Each component is normalized independently; the final score is their arithmetic mean. Higher \(S_{\mathrm{smooth}}\) indicates smoother generated motion. The metric is **official** when computation succeeds and requires no reference video.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `motion_smoothness`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/a0d0000e939d__image_dynamic_photorealistic_articulated_0136.jpg" alt="Input image: image_dynamic_photorealistic_articulated_0136" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.9546</div>
      <h4><code>image_dynamic_photorealistic_articulated_0136</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. A cheetah sprints across the dry grassy savannah, its legs cycling rapidly and body stretching low with each stride. Golden grass, sunlight, and shadows remain consistent around it.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: matrix-game-3-image-dynamic-5s image_dynamic_photorealistic_articulated_0136">
      <source src="../../../assets/metric_examples/b03e9f8572a3__matrix-game-3-image-dynamic-5s__image_dynamic_photorealistic_articulated_0136.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9557</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: longcat-video-image-dynamic image_dynamic_photorealistic_articulated_0136">
      <source src="../../../assets/metric_examples/c079d2132671__longcat-video-image-dynamic__image_dynamic_photorealistic_articulated_0136.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0011</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/88eeba3490ce__image_dynamic_photorealistic_deformable_0088.jpg" alt="Input image: image_dynamic_photorealistic_deformable_0088" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.9533</div>
      <h4><code>image_dynamic_photorealistic_deformable_0088</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. Coarse burlap fabric fills the frame, showing interlaced brown fibers and a rough woven pattern. The textile remains completely still, with no folds or fibers shifting.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: ac3d-image-dynamic image_dynamic_photorealistic_deformable_0088">
      <source src="../../../assets/metric_examples/5baf7f981cca__ac3d-image-dynamic__image_dynamic_photorealistic_deformable_0088.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9533</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: astra image_dynamic_photorealistic_deformable_0088">
      <source src="../../../assets/metric_examples/407c86893682__astra__image_dynamic_photorealistic_deformable_0088.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/bc6229f39d56__image_dynamic_stylized_fluid_0153.png" alt="Input image: image_dynamic_stylized_fluid_0153" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 0.9450</div>
      <h4><code>image_dynamic_stylized_fluid_0153</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. Water flows through the patterned canyon in wavy channels while smoke rises and curls from the canyon floor. Rock formations and green hills remain steady around the moving streams.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: matrix-game-3-image-dynamic-5s image_dynamic_stylized_fluid_0153">
      <source src="../../../assets/metric_examples/11d16fea29f8__matrix-game-3-image-dynamic-5s__image_dynamic_stylized_fluid_0153.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9450</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: sana-wm-image-dynamic image_dynamic_stylized_fluid_0153">
      <source src="../../../assets/metric_examples/bc33c7917ae4__sana-wm-image-dynamic__image_dynamic_stylized_fluid_0153.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/18d88e7e9117__image_dynamic_stylized_fluid_0083.png" alt="Input image: image_dynamic_stylized_fluid_0083" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 0.9402</div>
      <h4><code>image_dynamic_stylized_fluid_0083</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. The stylized waterfall flows over forest rocks in energetic swirls, sending water downward while colorful trees, foliage, and the nearby path remain still.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: ac3d-image-dynamic image_dynamic_stylized_fluid_0083">
      <source src="../../../assets/metric_examples/d2cd8d49f61b__ac3d-image-dynamic__image_dynamic_stylized_fluid_0083.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9494</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: astra image_dynamic_stylized_fluid_0083">
      <source src="../../../assets/metric_examples/5727ac3581d7__astra__image_dynamic_stylized_fluid_0083.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0092</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/0e26c207eead__image_dynamic_photorealistic_deformable_0085.jpg" alt="Input image: image_dynamic_photorealistic_deformable_0085" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 0.9360</div>
      <h4><code>image_dynamic_photorealistic_deformable_0085</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. In monochrome, dense wheat heads with long awns sway slightly, leaning and returning among overlapping stalks. The tight field view remains filled entirely with textured grain.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: ltx23-image-dynamic-8gpu-full_20260702_233041 image_dynamic_photorealistic_deformable_0085">
      <source src="../../../assets/metric_examples/7264ef47efca__ltx23-image-dynamic-8gpu-full_20260702_233041__image_dynamic_photorealistic_deformable_0085.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.9753</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: astra image_dynamic_photorealistic_deformable_0085">
      <source src="../../../assets/metric_examples/0d06447fc8b1__astra__image_dynamic_photorealistic_deformable_0085.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0393</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/5ba4124f8123__image_dynamic_stylized_deformable_0018.jpg" alt="Input image: image_dynamic_stylized_deformable_0018" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 0.8809</div>
      <h4><code>image_dynamic_stylized_deformable_0018</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. Leaves blow across the windy landscape as the two trees bend slightly under the force of the air. The muted sky and earthy ground remain steady behind the motion.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: sana-wm-image-dynamic image_dynamic_stylized_deformable_0018">
      <source src="../../../assets/metric_examples/6e26739d6044__sana-wm-image-dynamic__image_dynamic_stylized_deformable_0018.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.8809</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: cogvideo-x1.5-5b-i2v-image-dynamic image_dynamic_stylized_deformable_0018">
      <source src="../../../assets/metric_examples/53df8db0b16b__cogvideo-x1.5-5b-i2v-image-dynamic__image_dynamic_stylized_deformable_0018.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "VFIMamba: Video Frame Interpolation with State Space Models"

    Zhang et al., NeurIPS 2024.

!!! quote "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Zhang et al., CVPR 2018.
