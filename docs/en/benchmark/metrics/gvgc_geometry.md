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
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/ec72668a50dc__image_static_photorealistic_public_spaces_0064__input.jpg" alt="Input image: image_static_photorealistic_public_spaces_0064" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.0743</div>
      <h4><code>image_static_photorealistic_public_spaces_0064</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move left. A prison cell with barred door, cobwebs, and peeling paint.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_public_spaces_0064">
          <source src="../../../assets/metric_examples/curated_20260801/394f94fcdcda__image_static_photorealistic_public_spaces_0064__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9924</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>epipolar_error</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_public_spaces_0064">
          <source src="../../../assets/metric_examples/curated_20260801/448f0f8062ae__image_static_photorealistic_public_spaces_0064__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.9181</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>epipolar_error</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/a7645e0fbde8__image_static_photorealistic_living_spaces_0155__input.jpg" alt="Input image: image_static_photorealistic_living_spaces_0155" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.0770</div>
      <h4><code>image_static_photorealistic_living_spaces_0155</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A vintage bathroom with wooden shelves, bath products, and towels.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_living_spaces_0155">
          <source src="../../../assets/metric_examples/curated_20260801/adf26cec582b__image_static_photorealistic_living_spaces_0155__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9911</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>epipolar_error</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_living_spaces_0155">
          <source src="../../../assets/metric_examples/curated_20260801/c28e8366ed77__image_static_photorealistic_living_spaces_0155__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.9141</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>epipolar_error</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/bdb608aac636__image_static_stylized_living_spaces_0037__input.jpg" alt="Input image: image_static_stylized_living_spaces_0037" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.0860</div>
      <h4><code>image_static_stylized_living_spaces_0037</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A bedroom corner with wardrobe, chair, and rug.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
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
      <img src="../../../assets/metric_examples/curated_20260801/c298f05f69b4__image_static_stylized_public_spaces_0049__input.jpg" alt="Input image: image_static_stylized_public_spaces_0049" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.7193</div>
      <h4><code>image_static_stylized_public_spaces_0049</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. A library aisle with bookshelves, books, and reading table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
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
      <img src="../../../assets/metric_examples/curated_20260801/e1cb08434f22__image_static_photorealistic_public_spaces_0154__input.jpg" alt="Input image: image_static_photorealistic_public_spaces_0154" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.6446</div>
      <h4><code>image_static_photorealistic_public_spaces_0154</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. A library aisle with bookshelves, books, and reading tables.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_public_spaces_0154">
          <source src="../../../assets/metric_examples/curated_20260801/645f013eaa1d__image_static_photorealistic_public_spaces_0154__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9924</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_public_spaces_0154">
          <source src="../../../assets/metric_examples/curated_20260801/238c47aefdbf__image_static_photorealistic_public_spaces_0154__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3478</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/a896a0aef634__image_static_stylized_city_0157__input.jpg" alt="Input image: image_static_stylized_city_0157" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.6917</div>
      <h4><code>image_static_stylized_city_0157</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / outdoor / city</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit right. A voxel city street with brick buildings, crosswalk, and street signs.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_city_0157">
          <source src="../../../assets/metric_examples/curated_20260801/0c0c6f4e64c1__image_static_stylized_city_0157__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9927</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_city_0157">
          <source src="../../../assets/metric_examples/curated_20260801/2844ad2d0729__image_static_stylized_city_0157__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3010</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>epipolar_inlier_rate</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/14a72672e0d0__image_static_stylized_living_spaces_0134__input.jpg" alt="Input image: image_static_stylized_living_spaces_0134" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8146</div>
      <h4><code>image_static_stylized_living_spaces_0134</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit right. A modern living room with bookshelf, abstract painting, and sectional sofa.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0134">
          <source src="../../../assets/metric_examples/curated_20260801/356f3599300f__image_static_stylized_living_spaces_0134__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.8414</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0134">
          <source src="../../../assets/metric_examples/curated_20260801/e28be2531442__image_static_stylized_living_spaces_0134__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0268</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/7ee246743d64__image_static_stylized_living_spaces_0091__input.jpg" alt="Input image: image_static_stylized_living_spaces_0091" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8123</div>
      <h4><code>image_static_stylized_living_spaces_0091</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A bedroom corner with wardrobe, small bookshelf, and window.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0091">
          <source src="../../../assets/metric_examples/curated_20260801/7e539eca98ce__image_static_stylized_living_spaces_0091__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.8396</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0091">
          <source src="../../../assets/metric_examples/curated_20260801/1752059b5701__image_static_stylized_living_spaces_0091__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0273</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/624e8a1fef57__image_static_photorealistic_workspaces_0083__input.jpg" alt="Input image: image_static_photorealistic_workspaces_0083" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8311</div>
      <h4><code>image_static_photorealistic_workspaces_0083</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. An elegant office with bookshelf, books, and armchair.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_workspaces_0083">
          <source src="../../../assets/metric_examples/curated_20260801/ce09026b33ba__image_static_photorealistic_workspaces_0083__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.8699</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>feature_match_coverage</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_workspaces_0083">
          <source src="../../../assets/metric_examples/curated_20260801/e5b5f4c35b8e__image_static_photorealistic_workspaces_0083__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0388</div>
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
