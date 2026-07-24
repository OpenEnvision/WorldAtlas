# Prompt Alignment

Prompt Alignment measures **semantic agreement** between protocol anchor frames of a generated video and the target instruction \(T = \texttt{sample.prompt\_target}\). It evaluates whether the prediction realizes the requested scene content at benchmark-sampled time points, not whether the full rollout is temporally coherent or visually realistic.

Let \(\mathcal{A} = \{f_i\}_{i=1}^{M}\) denote the anchor-frame set selected by the evaluation protocol from the decoded prediction. With CLIP ViT-B/16 image encoder \(\phi_I\) and text encoder \(\phi_T\), define L2-normalized embeddings \(\mathbf{e}_i = \phi_I(f_i)\) and \(\mathbf{t} = \phi_T(T)\). The native CLIPScore for frame \(i\) is

\[
s_i = \max\!\left(100 \cdot \langle \mathbf{e}_i, \mathbf{t} \rangle,\, 0\right)
\tag{1}
\]

WorldArena maps each anchor score to \([0,1]\) via \(\tilde{s}_i = \mathrm{clamp}(s_i / 100,\, 0,\, 1)\) and reports the mean

\[
S_{\mathrm{prompt}} = \frac{1}{M}\sum_{i=1}^{M} \tilde{s}_i
\tag{2}
\]

Higher \(S_{\mathrm{prompt}}\) indicates stronger instruction following at protocol anchors. The metric is **official** when explicit instruction supervision is available and conditioning is not self-referential (`conditioning_strategy ≠ self`). Image suites may condition on a single reference frame, but the model prediction must still be a video.

The examples below are curated from the complete 2026-07-17 `image_static` snapshot: 20 models evaluated on the same 3,372 cases. Every side-by-side pair is strictly verified to share the sample ID, conditioning-image SHA-256, and actual generation-prompt SHA-256; both generation records must also identify that image as the conditioning input. “Good” and “poor” refer only to Prompt Alignment—not overall video quality—and each score is the mean over four `segment_random` protocol anchors.

<!-- metric-examples:start -->
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the actual generation prompt, then compares generated outputs. Every pair must match on sample_id, input-image SHA-256, and generation-prompt SHA-256. Cases are balanced across photorealistic/stylized, indoor/outdoor, scene categories, and score-gap bands instead of selecting only the largest gaps. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons. Prompt Alignment additionally requires both generation records to identify the displayed image as their actual conditioning input; outputs marked `.unused` or `no_image_conditioning` are excluded.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `prompt_alignment`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/066d09fa43d6__image_static_stylized_workspaces_0016.jpg" alt="Input image: image_static_stylized_workspaces_0016" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · actual same input + same prompt · score gap 0.1430</div>
      <h4><code>image_static_stylized_workspaces_0016</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / workspaces</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera pan right. A cozy workspace with desk, plants, and window. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: lyra-2-image-static-openenvision image_static_stylized_workspaces_0016">
      <source src="../../../assets/metric_examples/df6952a08217__lyra-2-image-static-openenvision__image_static_stylized_workspaces_0016.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.3210</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix_game2_5s_image_static image_static_stylized_workspaces_0016">
      <source src="../../../assets/metric_examples/cea803b80e21__matrix_game2_5s_image_static__image_static_stylized_workspaces_0016.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.1781</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/ba211409b845__image_static_photorealistic_passageways_0027.jpg" alt="Input image: image_static_photorealistic_passageways_0027" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · actual same input + same prompt · score gap 0.1399</div>
      <h4><code>image_static_photorealistic_passageways_0027</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera pan right. A dimly lit hallway with bulletin boards, side corridor, and fluorescent lights. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2 image_static_photorealistic_passageways_0027">
      <source src="../../../assets/metric_examples/ff54cba4f8aa__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_photorealistic_passageways_0027.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.3307</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix_game3_5s_image_static_dynamic image_static_photorealistic_passageways_0027">
      <source src="../../../assets/metric_examples/d4e2b61f96b4__matrix_game3_5s_image_static_dynamic__image_static_photorealistic_passageways_0027.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.1908</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/49bd12360acb__image_static_stylized_living_spaces_0053.jpg" alt="Input image: image_static_stylized_living_spaces_0053" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · actual same input + same prompt · score gap 0.1333</div>
      <h4><code>image_static_stylized_living_spaces_0053</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera push in. An attic bedroom with tree tops, sunlight, and large window. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: dreamx_world_image_static image_static_stylized_living_spaces_0053">
      <source src="../../../assets/metric_examples/82d6e3fe8cb2__dreamx_world_image_static__image_static_stylized_living_spaces_0053.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.2906</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: astra_image_static image_static_stylized_living_spaces_0053">
      <source src="../../../assets/metric_examples/e1dae4a903b7__astra_image_static__image_static_stylized_living_spaces_0053.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.1572</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/950d867a8d08__image_static_stylized_suburb_0052.jpg" alt="Input image: image_static_stylized_suburb_0052" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · actual same input + same prompt · score gap 0.1237</div>
      <h4><code>image_static_stylized_suburb_0052</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / suburb</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera move left. A rural landscape with hay bales, palm trees, and house. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: dreamx_world_image_static image_static_stylized_suburb_0052">
      <source src="../../../assets/metric_examples/8af8aa3e4606__dreamx_world_image_static__image_static_stylized_suburb_0052.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.3403</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2 image_static_stylized_suburb_0052">
      <source src="../../../assets/metric_examples/cd8122a8662b__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_stylized_suburb_0052.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2166</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/da7fa9fe82ca__image_static_photorealistic_living_spaces_0044.jpg" alt="Input image: image_static_photorealistic_living_spaces_0044" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · actual same input + same prompt · score gap 0.1167</div>
      <h4><code>image_static_photorealistic_living_spaces_0044</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / living_spaces</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera orbit right. A walk-in closet with wooden floor, white shelves, and wooden rods. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: dreamx_world_image_static image_static_photorealistic_living_spaces_0044">
      <source src="../../../assets/metric_examples/982b4dc1c557__dreamx_world_image_static__image_static_photorealistic_living_spaces_0044.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.3310</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2 image_static_photorealistic_living_spaces_0044">
      <source src="../../../assets/metric_examples/4b35c758e320__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_photorealistic_living_spaces_0044.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2143</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/32db1aa42523__image_static_stylized_city_0083.jpg" alt="Input image: image_static_stylized_city_0083" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · actual same input + same prompt · score gap 0.1047</div>
      <h4><code>image_static_stylized_city_0083</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / city</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera pan right. A city park with swings, children, and playground. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2 image_static_stylized_city_0083">
      <source src="../../../assets/metric_examples/0a2b92a949e9__matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2__image_static_stylized_city_0083.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.2856</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: astra_image_static image_static_stylized_city_0083">
      <source src="../../../assets/metric_examples/a0b221262c5e__astra_image_static__image_static_stylized_city_0083.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.1809</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/bee6392148ee__image_static_photorealistic_verdant_landscape_0048.jpeg" alt="Input image: image_static_photorealistic_verdant_landscape_0048" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 7 · actual same input + same prompt · score gap 0.0999</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0048</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera pan left. A rice field with palm trees, small farmhouse, and dirt path. Everything in the scene remains completely motionless.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: matrix_game3_5s_image_static_dynamic image_static_photorealistic_verdant_landscape_0048">
      <source src="../../../assets/metric_examples/cb08f9578717__matrix_game3_5s_image_static_dynamic__image_static_photorealistic_verdant_landscape_0048.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.3020</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: astra_image_static image_static_photorealistic_verdant_landscape_0048">
      <source src="../../../assets/metric_examples/e77772bf668f__astra_image_static__image_static_photorealistic_verdant_landscape_0048.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.2021</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/52fa238c06b3__image_static_photorealistic_suburb_0128.jpg" alt="Input image: image_static_photorealistic_suburb_0128" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 8 · actual same input + same prompt · score gap 0.0996</div>
      <h4><code>image_static_photorealistic_suburb_0128</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / suburb</span><span>Actual input: verified</span></div>
      <div class="metric-case-prompt">
        <span>Generation prompt</span>
        <p>Camera pull out. A suburban street with houses, streetlights, and parked cars.</p>
        <small>prompt source: generation_records</small>
      </div>
</div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: ltx23-image-static-8gpu-full_official_default_20260613_031436 image_static_photorealistic_suburb_0128">
      <source src="../../../assets/metric_examples/1fe3f1c037ba__ltx23-image-static-8gpu-full_official_default_20260613_031436__image_static_photorealistic_suburb_0128.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 0.2862</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Good output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: ac3d-image-static image_static_photorealistic_suburb_0128">
      <source src="../../../assets/metric_examples/2812bd119578__ac3d-image-static__image_static_photorealistic_suburb_0128.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.1865</div>
  </div>
  <figcaption class="metric-output-caption">
    <strong>Poor output</strong>
    <span>Metric: prompt_alignment · protocol: segment_random</span>
  </figcaption>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
