# Temporal Flickering

Temporal Flickering quantifies **inter-frame pixel stability** in a generated video. High-frequency brightness or color shifts between consecutive frames produce visible flicker; this metric penalizes such instability using mean absolute error (MAE) on RGB pixels.

Given $N$ decoded frames $\{x_i\}_{i=1}^{N}$, the per-pair pixel MAE is

\[
\mathrm{MAE}(x_i, x_{i+1})
=
\frac{1}{HWC}\sum_{h,w,c}\left|x_i(h,w,c)-x_{i+1}(h,w,c)\right|
\tag{1}
\]

where $H$, $W$, and $C$ are height, width, and channel count. WorldArena evaluates MAE on uint8 RGB values in $[0,255]$. When resolutions differ, the later frame is resized to match the earlier frame before differencing.

The temporal flickering score aggregates adjacent-frame MAE:

\[
S_{\mathrm{flick}}
=
\frac{255 - \frac{1}{N-1}\sum_{i=1}^{N-1}\mathrm{MAE}(x_i, x_{i+1})}{255}
\tag{2}
\]

Higher $S_{\mathrm{flick}}$ indicates less flicker. The metric is **prediction-only** and requires at least two frames.

<!-- temporal-flickering-representative-cases:start -->
## Representative Temporal Flickering Comparisons

These representative comparisons were selected from the official `image_static` Temporal Flickering results and manually reviewed as frame strips. Each exhibit fixes one benchmark case—input image and eval-plan prompt target—and compares two prediction videos. Temporal Flickering is higher-is-better; mean pairwise pixel MAE is lower-is-better.

Adapter-specific conditioning may differ, so these are metric diagnostics rather than controlled model rankings. Exact paths, hashes, frame-pair statistics, and report provenance are in the [machine-readable Temporal Flickering manifest](../../../assets/metric_examples/temporal_flickering_representative_manifest.csv).

<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/5cd19d15663a__image_static_photorealistic_aquatic_landscape_0098.jpg" alt="Input image: image_static_photorealistic_aquatic_landscape_0098" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · visible instability increases adjacent-frame MAE</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0098</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. An aquatic landscape with pebble beach, calm sea, and blue sky.</p>
      </div>
      <p class="metric-case-note">Output A changes smoothly. Output B develops large view distortions and moving black borders, producing much larger adjacent-frame pixel differences.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output A: ac3d-image-static, aquatic landscape 0098">
          <source src="../../../assets/metric_examples/d6290f43f723__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output A</strong>
        <span>Temporal Flickering: <strong>0.9812</strong> · mean pairwise MAE 4.7833</span>
      </figcaption>
    </figure>
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output B: LucidDreamer, aquatic landscape 0098">
          <source src="../../../assets/metric_examples/20dc16268a49__luciddreamer-image-static-8gpu-20260613_151844__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output B</strong>
        <span>Temporal Flickering: <strong>0.7074</strong> · mean pairwise MAE 74.6172</span>
      </figcaption>
    </figure>
  </div>
</article>

<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f6e5fd26f831__image_static_photorealistic_workspaces_0106.jpg" alt="Input image: image_static_photorealistic_workspaces_0106" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · a nearly static output can score very highly</div>
      <h4><code>image_static_photorealistic_workspaces_0106</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move left. A workspaces scene with bookshelf, indoor plants, and adjacent wall.</p>
      </div>
      <p class="metric-case-note">Output B is nearly static and therefore scores 0.9940, even though it does not preserve the reference workspace. Temporal Flickering measures only prediction-side pixel stability; it does not check scene correctness or conditioning fidelity.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output A: ac3d-image-static, workspace 0106">
          <source src="../../../assets/metric_examples/c0b5d552bebf__ac3d-image-static__image_static_photorealistic_workspaces_0106.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output A</strong>
        <span>Temporal Flickering: <strong>0.9769</strong> · mean pairwise MAE 5.8888</span>
      </figcaption>
    </figure>
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output B: CameraCtrl, workspace 0106">
          <source src="../../../assets/metric_examples/5a90ac712030__cameractrl_image_static__image_static_photorealistic_workspaces_0106.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output B</strong>
        <span>Temporal Flickering: <strong>0.9940</strong> · mean pairwise MAE 1.5278</span>
      </figcaption>
    </figure>
  </div>
</article>

<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/c857427716ae__image_static_photorealistic_living_spaces_0140.jpg" alt="Input image: image_static_photorealistic_living_spaces_0140" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · adjacent-pixel MAE is motion-sensitive</div>
      <h4><code>image_static_photorealistic_living_spaces_0140</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera push in. A living room with sofa, bookshelf, and books.</p>
      </div>
      <p class="metric-case-note">Output A moves little, while Output B makes a larger smooth camera move and reveals the bookshelf. The lower score for Output B reflects larger pixel changes; by itself it does not prove visible high-frequency flicker because this implementation is not motion-compensated.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output A: ac3d-image-static, living space 0140">
          <source src="../../../assets/metric_examples/3a76d4fa805b__ac3d-image-static__image_static_photorealistic_living_spaces_0140.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output A</strong>
        <span>Temporal Flickering: <strong>0.9634</strong> · mean pairwise MAE 9.3268</span>
      </figcaption>
    </figure>
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output B: LongCat-Video, living space 0140">
          <source src="../../../assets/metric_examples/d43209bc4d20__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_living_spaces_0140.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output B</strong>
        <span>Temporal Flickering: <strong>0.7694</strong> · mean pairwise MAE 58.8134</span>
      </figcaption>
    </figure>
  </div>
</article>

</div>
<!-- temporal-flickering-representative-cases:end -->

## References {#reference}

!!! quote "VBench — Temporal Flickering (Appendix C.2.3)"

    Consecutive-frame pixel MAE converted to a $[0,1]$ stability score via $(255 - \overline{\mathrm{MAE}})/255$.
