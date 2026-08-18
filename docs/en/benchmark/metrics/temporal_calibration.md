# Temporal Calibration

Temporal Calibration measures whether a generated video preserves a **stable physical clock**. When a file declares a playback frame rate, does the motion visible in the frames imply the same real-time progression? WorldArena applies this as an output-only audit in the `real_time` dimension using the wrapped Pulse-of-Motion / Visual Chronometer predictor [[Gao et al., 2026](#reference-visual-chronometer)].

This is not the same as optical flow, motion magnitude, or temporal smoothness. A rollout can look smooth frame to frame and still be temporally wrong: falls may drift as if gravity were slowed, or fast periodic motion may appear unnaturally sluggish. Visual Chronometer calls this failure mode **chronometric hallucination**—visually coherent dynamics with an ambiguous, unstable, or uncontrollable physical time scale.

The predictor estimates **Physical FPS** (`PhyFPS`) from overlapping temporal windows instead of trusting container metadata alone. WorldArena compares the mean inferred rate with the nominal playback FPS $f_{\mathrm{meta}}$ (OpenCV container FPS, or manifest fallback) and reports three official metrics: `physical_fps_error`, `physical_fps_pct_error`, and `physical_fps_intra_video_cv`. The first two capture global tempo mismatch; the third captures rhythm drift inside a single clip.

For each prediction, the runtime extracts overlapping windows with defaults `clip_length = 30`, `stride = 4`, and `resolution = 216`. Each window yields a segment estimate $\hat{f}_k$. The mean inferred physical FPS is

\[
\bar{f}_{\mathrm{phy}}
=
\frac{1}{K}\sum_{k=1}^{K} \hat{f}_k
\tag{1}
\]

Absolute and relative calibration errors are

\[
E_{\mathrm{abs}} = \left|\bar{f}_{\mathrm{phy}} - f_{\mathrm{meta}}\right|
\tag{2}
\]

\[
E_{\mathrm{pct}} = \frac{E_{\mathrm{abs}}}{f_{\mathrm{meta}}}
\tag{3}
\]

Within-video instability is the coefficient of variation over segment predictions:

\[
C_{\mathrm{intra}}
=
\frac{\operatorname{std}(\hat{f}_1,\ldots,\hat{f}_K)}{\operatorname{mean}(\hat{f}_1,\ldots,\hat{f}_K)}
\tag{4}
\]

WorldArena maps `physical_fps_error`, `physical_fps_pct_error`, and `physical_fps_intra_video_cv` to $[0,1]$ with fixed reverse scaling (`physical_fps_error`: $[0,30]$; the other two: $[0,1]$). **Higher normalized values mean better temporal calibration.** When the side runtime, checkpoint, or FPS metadata is unavailable, the metric family is reported as `not_applicable`.

## References {#reference-visual-chronometer}

!!! quote "The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"

    Xiangbo Gao, Mingyang Wu, Siyuan Yang, Jiongze Yu, Pardis Taghavi, Fangzhou Lin, and Zhengzhong Tu.  
    arXiv:2603.14375, 2026.

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/b7d0e735d01a__image_static_stylized_verdant_landscape_0159__input.jpg" alt="Input image: image_static_stylized_verdant_landscape_0159" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9733</div>
      <h4><code>image_static_stylized_verdant_landscape_0159</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. A stylized courtyard with stone pathway, plants, and adjacent garden.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_verdant_landscape_0159">
          <source src="../../../assets/metric_examples/curated_20260801/d10703dd2dfa__image_static_stylized_verdant_landscape_0159__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9867</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_error</span><span>Estimated physical FPS 15.4 · declared FPS 15.0 · absolute error 0.4</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_verdant_landscape_0159">
          <source src="../../../assets/metric_examples/curated_20260801/641547bfe52f__image_static_stylized_verdant_landscape_0159__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0133</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_error</span><span>Estimated physical FPS 45.6 · declared FPS 16.0 · absolute error 29.6</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c4d9230ba2a0__image_static_photorealistic_city_0047__input.jpg" alt="Input image: image_static_photorealistic_city_0047" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9700</div>
      <h4><code>image_static_photorealistic_city_0047</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan left. A historic city street with adjacent street, historic buildings, and pedestrians.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_city_0047">
          <source src="../../../assets/metric_examples/curated_20260801/616843ed9241__image_static_photorealistic_city_0047__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9733</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_error</span><span>Estimated physical FPS 24.8 · declared FPS 24.0 · absolute error 0.8</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_city_0047">
          <source src="../../../assets/metric_examples/curated_20260801/485cdc128fad__image_static_photorealistic_city_0047__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0033</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_error</span><span>Estimated physical FPS 37.9 · declared FPS 8.0 · absolute error 29.9</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/77047b98a662__image_static_photorealistic_dining_spaces_0061__input.jpg" alt="Input image: image_static_photorealistic_dining_spaces_0061" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9800</div>
      <h4><code>image_static_photorealistic_dining_spaces_0061</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pull out. A bar interior with seating area, tables, and patrons.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_dining_spaces_0061">
          <source src="../../../assets/metric_examples/curated_20260801/77d04c8a566c__image_static_photorealistic_dining_spaces_0061__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9867</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_error</span><span>Estimated physical FPS 60.4 · declared FPS 60.0 · absolute error 0.4</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_dining_spaces_0061">
          <source src="../../../assets/metric_examples/curated_20260801/d82b8808bafe__image_static_photorealistic_dining_spaces_0061__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0067</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_error</span><span>Estimated physical FPS 44.8 · declared FPS 15.0 · absolute error 29.8</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/9520a96c9ee1__image_static_stylized_city_0137__input.jpg" alt="Input image: image_static_stylized_city_0137" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.6648</div>
      <h4><code>image_static_stylized_city_0137</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A japanese storefront with adjacent storefront, street vendor, and pedestrians.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_city_0137">
          <source src="../../../assets/metric_examples/curated_20260801/acb2c2032abe__image_static_stylized_city_0137__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9840</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_intra_video_cv</span><span>Segment physical FPS 60.1–63.0 · coefficient of variation 0.016</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_city_0137">
          <source src="../../../assets/metric_examples/curated_20260801/781bbd8a087a__image_static_stylized_city_0137__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3192</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_intra_video_cv</span><span>Segment physical FPS 12.8–55.8 · coefficient of variation 0.681</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c464d64b3556__image_static_stylized_aquatic_landscape_0148__input.jpg" alt="Input image: image_static_stylized_aquatic_landscape_0148" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.6596</div>
      <h4><code>image_static_stylized_aquatic_landscape_0148</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A coastal landscape with rocky outcrop, boats, and extended shoreline.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_aquatic_landscape_0148">
          <source src="../../../assets/metric_examples/curated_20260801/de99760ac54c__image_static_stylized_aquatic_landscape_0148__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9788</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_intra_video_cv</span><span>Segment physical FPS 19.3–21.0 · coefficient of variation 0.021</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_aquatic_landscape_0148">
          <source src="../../../assets/metric_examples/curated_20260801/4c25fd4af6e3__image_static_stylized_aquatic_landscape_0148__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3192</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_intra_video_cv</span><span>Segment physical FPS 11.6–47.6 · coefficient of variation 0.681</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/feb41abcc29d__image_static_photorealistic_workspaces_0144__input.jpg" alt="Input image: image_static_photorealistic_workspaces_0144" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.7192</div>
      <h4><code>image_static_photorealistic_workspaces_0144</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move left. A modern conference room with side wall, whiteboard, and chairs. Camera pull out. A modern conference room with entrance door, coat rack, and chairs. Camera pan left. A modern conference room with hallway and artwork.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_workspaces_0144">
          <source src="../../../assets/metric_examples/curated_20260801/61c84b8ed7cc__image_static_photorealistic_workspaces_0144__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9894</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_intra_video_cv</span><span>Segment physical FPS 69.1–71.7 · coefficient of variation 0.011</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_workspaces_0144">
          <source src="../../../assets/metric_examples/curated_20260801/ed03c3e72ce1__image_static_photorealistic_workspaces_0144__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2702</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_intra_video_cv</span><span>Segment physical FPS 11.9–54.8 · coefficient of variation 0.730</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/08d3e1c44186__image_static_photorealistic_passageways_0139__input.jpg" alt="Input image: image_static_photorealistic_passageways_0139" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9983</div>
      <h4><code>image_static_photorealistic_passageways_0139</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move left. A walk-in closet with shelves, clothing racks, and clothing.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0139">
          <source src="../../../assets/metric_examples/curated_20260801/564b831d08d3__image_static_photorealistic_passageways_0139__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9983</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_pct_error</span><span>Estimated physical FPS 59.9 · declared FPS 60.0 · relative error 0.17%</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0139">
          <source src="../../../assets/metric_examples/curated_20260801/191ee5f9e93e__image_static_photorealistic_passageways_0139__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_pct_error</span><span>Estimated physical FPS 40.0 · declared FPS 15.0 · relative error 166.67%</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/72f1d4727577__image_static_photorealistic_public_spaces_0167__input.jpg" alt="Input image: image_static_photorealistic_public_spaces_0167" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9983</div>
      <h4><code>image_static_photorealistic_public_spaces_0167</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera push in. A circular room with marble pedestal, intricate floor pattern, and glowing base.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_public_spaces_0167">
          <source src="../../../assets/metric_examples/curated_20260801/3c85cdec6032__image_static_photorealistic_public_spaces_0167__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9983</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_pct_error</span><span>Estimated physical FPS 60.1 · declared FPS 60.0 · relative error 0.17%</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_public_spaces_0167">
          <source src="../../../assets/metric_examples/curated_20260801/54a5bf98aaae__image_static_photorealistic_public_spaces_0167__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_pct_error</span><span>Estimated physical FPS 50.4 · declared FPS 15.0 · relative error 236.00%</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/3d5e5f5585e3__image_static_photorealistic_passageways_0115__input.jpg" alt="Input image: image_static_photorealistic_passageways_0115" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.9983</div>
      <h4><code>image_static_photorealistic_passageways_0115</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit right. A dimly lit alleyway with upper landing, doorway, and railing.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0115">
          <source src="../../../assets/metric_examples/curated_20260801/48b996e1b9e0__image_static_photorealistic_passageways_0115__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9983</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>physical_fps_pct_error</span><span>Estimated physical FPS 60.1 · declared FPS 60.0 · relative error 0.17%</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0115">
          <source src="../../../assets/metric_examples/curated_20260801/f32b8752d3c3__image_static_photorealistic_passageways_0115__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>physical_fps_pct_error</span><span>Estimated physical FPS 41.0 · declared FPS 15.0 · relative error 173.33%</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
