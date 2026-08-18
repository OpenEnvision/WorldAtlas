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
## Verified Same-Input Temporal Flickering Example

This page keeps one case only. Both video first frames pass geometric matching against the displayed input; scores illustrate this metric only, not overall model ranking.

[Machine-readable audit manifest](../../../assets/metric_examples/temporal_flickering_representative_manifest.csv).

<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/c857427716ae__image_static_photorealistic_living_spaces_0140.jpg" alt="Input image: image_static_photorealistic_living_spaces_0140" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Verified case · adjacent-pixel MAE is motion-sensitive</div>
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
