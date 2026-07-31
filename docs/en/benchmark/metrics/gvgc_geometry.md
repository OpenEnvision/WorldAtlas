# Epipolar Geometry

This page defines three **GVGC-style epipolar-geometry diagnostic metrics** that probe whether nearby generated frames admit a stable two-view epipolar explanation [[Hartley and Zisserman, 2004](#reference)]. They require only the prediction video—no camera-pose annotations and no reference video. The core question is whether local feature correspondences approximately satisfy the epipolar constraint, indicating that two frames can be explained as projections of the same scene under a pinhole camera model with rigid relative motion.

In classical multiple-view geometry, if two frames observe the same scene under rigid camera motion, a point \(x\) in the first frame must map to a corresponding point \(x'\) that lies on the epipolar line \(l' = F x\) induced by the fundamental matrix \(F\). Generated videos with background warping, local object drift, unstable depth layering, or frame pairs that are not projections of one 3D structure will struggle to maintain many RANSAC inliers under this constraint, raising Sampson error and lowering inlier rate. These metrics therefore provide a **lightweight, interpretable** two-view geometry signal; they check sparse-feature epipolar consistency only and do not replace global 3D self-consistency metrics such as MEt3R, reprojection error, or reconstruction-based scores.

In WorldArena, the backend builds adjacent (or strided) frame pairs from sampled prediction frames, matches SIFT features with Lowe's ratio test by default, and estimates \(F\) with OpenCV RANSAC. It computes Sampson error on RANSAC inliers and reports inlier rate and coverage; if the primary path fails, the runtime may fall back to relaxed matching thresholds or Shi-Tomasi + Lucas-Kanade tracking. All three sub-metrics share one backend pass:

| Metric | Raw statistic | Direction |
| --- | --- | --- |
| `epipolar_error` | median inlier Sampson error after fundamental-matrix RANSAC | lower is better |
| `epipolar_inlier_rate` | RANSAC inliers / Lowe-ratio feature matches | higher is better |
| `feature_match_coverage` | RANSAC inliers / min keypoint count | higher is better |

For matched homogeneous points \(x_i, x'_i\), Sampson error is

\[
d_i = \frac{|{x'_i}^{\mathsf T} F x_i|}{\sqrt{(Fx_i)_1^2 + (Fx_i)_2^2 + (F^{\mathsf T}x'_i)_1^2 + (F^{\mathsf T}x'_i)_2^2}}
\tag{1}
\]

Per valid pair, `epipolar_error` uses the median \(d_i\) over RANSAC inliers; the video-level value is the median across pairs. Normalization maps \([0, 5]\) with lower-is-better scaling.

Per valid pair, inlier rate and coverage are

\[
r = \frac{\#\mathrm{inliers}}{\#\mathrm{good\ matches}},
\qquad
c = \min\!\left(\frac{\#\mathrm{inliers}}{\min(\#\mathrm{kp}_\mathrm{left}, \#\mathrm{kp}_\mathrm{right})},\, 1\right)
\tag{2}
\]

Video-level `epipolar_inlier_rate` and `feature_match_coverage` average \(r\) and \(c\) over valid pairs with identity normalization on \([0,1]\). `epipolar_error` takes the median of per-pair medians and is reverse-normalized from \([0, 5]\) to \([0, 1]\) (higher normalized score means stronger epipolar consistency). If no pair yields sufficient inliers, all three metrics return `not_applicable`. They are currently **diagnostic**: reported separately and excluded from headline aggregation. They may fail often in weak-texture, strongly non-rigid, or dynamic-prompt regimes and should be read alongside MEt3R and reprojection error.

<!-- metric-examples:start -->
## Human-reviewed Same-Case Example

The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/bdb608aac636__image_static_stylized_living_spaces_0037.jpg" alt="Input image: image_static_stylized_living_spaces_0037" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.0860</div>
      <h4><code>image_static_stylized_living_spaces_0037</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera pan right. A bedroom corner with wardrobe, chair, and rug.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring camera move retains recoverable scene geometry; the lower-scoring rollout progressively darkens and loses cross-view correspondences.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0037">
          <source src="../../../assets/metric_examples/curated_20260801/7f6f8eb8d3f9__image_static_stylized_living_spaces_0037__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9871</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>epipolar_error</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0037">
          <source src="../../../assets/metric_examples/c223556eb328__matrix_game2_5s_image_static__image_static_stylized_living_spaces_0037.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.9011</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>epipolar_error</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/116e66680884__image_static_stylized_public_spaces_0049__input.jpg" alt="Input image: image_static_stylized_public_spaces_0049" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.7193</div>
      <h4><code>image_static_stylized_public_spaces_0049</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera orbit left. A library aisle with bookshelves, books, and reading table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout keeps the aisle layout trackable across the orbit; the lower-scoring rollout breaks the shelf geometry into incompatible views.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_public_spaces_0049">
          <source src="../../../assets/metric_examples/curated_20260801/080234cb97e9__image_static_stylized_public_spaces_0049__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9841</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_public_spaces_0049">
          <source src="../../../assets/metric_examples/curated_20260801/d609d31a9446__image_static_stylized_public_spaces_0049__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2648</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/32d68cb087eb__image_static_photorealistic_passageways_0155.jpg" alt="Input image: image_static_photorealistic_passageways_0155" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Human reviewed · same case, input and prompt · gap 0.8407</div>
      <h4><code>image_static_photorealistic_passageways_0155</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Exact generation prompt</span>
        <p>Camera push in. A concrete corridor with end wall and light source.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">The higher-scoring rollout preserves trackable corridor structure across the frame; the lower-scoring rollout collapses into texture-only walls with poor spatial coverage.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0155">
          <source src="../../../assets/metric_examples/35ab3fc3e20b__cameractrl_image_static__image_static_photorealistic_passageways_0155.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.8889</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0155">
          <source src="../../../assets/metric_examples/bdf8a95737dc__matrix_game3_5s_image_static_dynamic__image_static_photorealistic_passageways_0155.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0482</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## References {#reference}

!!! quote "Multiple View Geometry in Computer Vision"

    Hartley & Zisserman, Cambridge University Press, 2004.
