# Motion Magnitude

Motion Magnitude quantifies **how much internal motion** a generated video contains, independent of motion correctness. It detects conservative generation—visually stable but nearly static outputs—without evaluating direction, semantics, or object behavior.

For adjacent frames \((I_t, I_{t+1})\), SEA-RAFT yields flow magnitude \(F_t(x,y) = \sqrt{u_t^2 + v_t^2}\). The frame-pair statistic is the pixel median; the raw score averages over pairs:

\[
S_{\mathrm{mag}}^{\mathrm{raw}} = \frac{1}{T-1}\sum_{t=0}^{T-2} \mathrm{median}_{x,y}\, F_t(x,y)
\tag{1}
\]

WorldArena z-score calibrates with \(\mu = 3.2425\), \(\sigma = 3.4505\), clip \([-0.8638, 2.7498]\), mapping to \([0.25, 0.75]\). Higher normalized scores indicate stronger visible motion. The metric is **official** when computation succeeds.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `motion_magnitude`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/10d5b411a824__image_dynamic_photorealistic_deformable_0090.jpg" alt="Input image: image_dynamic_photorealistic_deformable_0090" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 0.7605</div>
      <h4><code>image_dynamic_photorealistic_deformable_0090</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. The blue denim fabric subtly wrinkles and stretches, shifting the weave pattern as a fold travels across the surface. The close-up lighting and textile background stay consistent.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: yume-sample5b-image-dynamic-5s image_dynamic_photorealistic_deformable_0090">
      <source src="../../../assets/metric_examples/497f3ecfb716__yume-sample5b-image-dynamic-5s__image_dynamic_photorealistic_deformable_0090.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: hunyuan-video-i2v-image-static-dynamic image_dynamic_photorealistic_deformable_0090">
      <source src="../../../assets/metric_examples/253f174d78d3__hunyuan-video-i2v-image-static-dynamic__image_dynamic_photorealistic_deformable_0090.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2395</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/167a8b97141d__image_dynamic_photorealistic_deformable_0089.jpg" alt="Input image: image_dynamic_photorealistic_deformable_0089" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 0.7605</div>
      <h4><code>image_dynamic_photorealistic_deformable_0089</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. Green wheat stalks with unripe ears sway forward and spring back, their leaves fluttering in the breeze. The soft background of the field remains visually stable.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: astra image_dynamic_photorealistic_deformable_0089">
      <source src="../../../assets/metric_examples/7775b860f58b__astra__image_dynamic_photorealistic_deformable_0089.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix_game2_5s image_dynamic_photorealistic_deformable_0089">
      <source src="../../../assets/metric_examples/d0f5d3df7a68__matrix_game2_5s__image_dynamic_photorealistic_deformable_0089.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2395</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/c64d7df96bb7__image_dynamic_photorealistic_deformable_0064.jpg" alt="Input image: image_dynamic_photorealistic_deformable_0064" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 0.7605</div>
      <h4><code>image_dynamic_photorealistic_deformable_0064</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. Golden barley stalks sway gently in sunlight, their fine strands bending and shimmering as a light breeze passes through. The field remains dense and continuous around them.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: longcat-video-image-dynamic image_dynamic_photorealistic_deformable_0064">
      <source src="../../../assets/metric_examples/d7eba57786ee__longcat-video-image-dynamic__image_dynamic_photorealistic_deformable_0064.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix_game2_5s image_dynamic_photorealistic_deformable_0064">
      <source src="../../../assets/metric_examples/986a22fd3c5d__matrix_game2_5s__image_dynamic_photorealistic_deformable_0064.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2395</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/73a085d1c822__image_dynamic_photorealistic_deformable_0163.jpg" alt="Input image: image_dynamic_photorealistic_deformable_0163" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 0.7604</div>
      <h4><code>image_dynamic_photorealistic_deformable_0163</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. Translucent fabric layers ripple in a gentle swirling motion, their curved folds sliding over one another. The close-up composition stays focused on the delicate overlapping textures.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: astra image_dynamic_photorealistic_deformable_0163">
      <source src="../../../assets/metric_examples/42143df67a10__astra__image_dynamic_photorealistic_deformable_0163.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: ac3d-image-dynamic image_dynamic_photorealistic_deformable_0163">
      <source src="../../../assets/metric_examples/aba224b60229__ac3d-image-dynamic__image_dynamic_photorealistic_deformable_0163.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2396</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/3929c1e0d708__image_dynamic_photorealistic_articulated_0068.jpg" alt="Input image: image_dynamic_photorealistic_articulated_0068" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 0.7604</div>
      <h4><code>image_dynamic_photorealistic_articulated_0068</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. The dog runs quickly across the grassy field, moving forward with alternating legs, open mouth, and perked ears while the fence and trees stay in place.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: longcat-video-image-dynamic image_dynamic_photorealistic_articulated_0068">
      <source src="../../../assets/metric_examples/1747010395f4__longcat-video-image-dynamic__image_dynamic_photorealistic_articulated_0068.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix_game2_5s image_dynamic_photorealistic_articulated_0068">
      <source src="../../../assets/metric_examples/aff317072a9d__matrix_game2_5s__image_dynamic_photorealistic_articulated_0068.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2396</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/fe16fdec2c52__image_dynamic_photorealistic_articulated_0001.jpg" alt="Input image: image_dynamic_photorealistic_articulated_0001" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 0.7604</div>
      <h4><code>image_dynamic_photorealistic_articulated_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera fixed. A bird of prey glides forward low over a grassy field, wings spread wide with subtle body and wing adjustments. The wooden fence post, wild grasses, and vines remain still.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: astra image_dynamic_photorealistic_articulated_0001">
      <source src="../../../assets/metric_examples/73ee88755197__astra__image_dynamic_photorealistic_articulated_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: sana-wm-image-dynamic image_dynamic_photorealistic_articulated_0001">
      <source src="../../../assets/metric_examples/3c22680fb047__sana-wm-image-dynamic__image_dynamic_photorealistic_articulated_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2396</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., ECCV 2024.
