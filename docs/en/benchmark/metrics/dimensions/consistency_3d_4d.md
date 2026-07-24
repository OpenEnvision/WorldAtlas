# 3D/4D Consistency

<p class="dimension-lede">The <code>consistency_3d_4d</code> dimension evaluates whether a generated rollout admits a <strong>stable spatiotemporal geometric explanation</strong> in three or four dimensions. Unlike quality metrics that judge single-frame plausibility or low-level appearance stability, this dimension asks whether camera motion, depth layering, cross-frame correspondence, newly exposed regions, and motion fields can be explained jointly under one geometric hypothesis across the full video.</p>

<div class="dimension-overview dimension-overview--consistency_3d_4d">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">Dimension key</div>
  <div class="dimension-hero-value"><code>consistency_3d_4d</code></div>
  <div class="dimension-hero-note">Rolls up into <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Metric families</div>
  <div class="dimension-hero-value">6</div>
  <div class="dimension-hero-note">Novel regions, multiview, epipolar, reprojection, depth, motion</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Core question</div>
  <div class="dimension-hero-value dimension-hero-value--compact">Geometric coherence</div>
  <div class="dimension-hero-note">Does the video evolve like a continuous viewpoint in a 3D world?</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Metric families and geometric semantics</p>
<p class="dimension-section-note">Readings organized from visible regions to newly exposed regions, local epipolar constraints to global reconstruction, depth structure to short-horizon motion fields.</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">Novel regions</div>
  <p class="dimension-family-note">Masked appearance fidelity when the reference camera path exposes regions unseen in the first frame.</p>
  <ul class="dimension-family-metrics">
    <li><a href="../novel_region_fidelity.md">novel_region_fidelity</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Multiview / 3D representation</div>
  <p class="dimension-family-note">Consistency between anchor viewpoints and whether the prediction can be reconstructed as 3D Gaussians and re-rendered along the view path.</p>
  <ul class="dimension-family-metrics">
    <li><a href="../met3r_consistency.md">met3r_consistency</a></li>
    <li><a href="../reconstruction_consistency.md">reconstruction_consistency</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Classical epipolar geometry</div>
  <p class="dimension-family-note">Whether adjacent frames admit a stable fundamental-matrix / Sampson epipolar explanation.</p>
  <ul class="dimension-family-metrics">
    <li><a href="../gvgc_geometry.md#epipolar-error">epipolar_error</a></li>
    <li><a href="../gvgc_geometry.md#epipolar-inlier-rate">epipolar_inlier_rate</a></li>
    <li><a href="../gvgc_geometry.md#feature-match-coverage">feature_match_coverage</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Camera and reprojection</div>
  <p class="dimension-family-note">Whether dense SLAM-recovered poses and depths reproject covisible points consistently onto the image plane.</p>
  <ul class="dimension-family-metrics">
    <li><a href="../reprojection_error.md">reprojection_error</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Depth structure</div>
  <p class="dimension-family-note">Whether relative depth ordering is preserved and whether the forward viewing corridor shows implausible near geometry.</p>
  <ul class="dimension-family-metrics">
    <li><a href="../depth_accuracy.md">depth_accuracy</a></li>
    <li><a href="../depth_collision.md">depth_collision</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Motion field</div>
  <p class="dimension-family-note">Short-horizon photometric self-consistency under forward–backward optical-flow round-trips.</p>
  <ul class="dimension-family-metrics">
    <li><a href="../optical_flow_aepe.md">optical_flow_aepe</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-body">

### Formal Reading

For a given sample and formal suite, let $\mathcal{M}_{\mathrm{off}}$ denote the set of **official** metrics under `consistency_3d_4d` that are successfully computed. Each metric $m \in \mathcal{M}_{\mathrm{off}}$ yields a normalized score $\tilde{s}_m \in [0,1]$ with the WorldArena convention that **higher is better**. The suite-level dimension summary is the arithmetic mean over available official metrics:

\[
S_{\mathrm{3D/4D}}
=
\frac{1}{|\mathcal{M}_{\mathrm{off}}|}
\sum_{m \in \mathcal{M}_{\mathrm{off}}}
\tilde{s}_m
\tag{1}
\]

Equation (1) aggregates **official** readings only. Non-official metrics provide complementary geometric probes but do not enter `per_dimension_summary.json`. If an official metric returns `normalized = null` because of missing inputs, reconstruction failure, or unmet eligibility constraints, it is excluded from Eq. (1) and is not automatically treated as zero.

In terms of constraint strength, the epipolar trio provides the **lightest** two-view consistency signal and is most reliable for textured, mostly static scenes with camera motion. `reprojection_error` and reconstruction-style metrics require the prediction to support a **dense 3D explanation over the full trajectory**. `novel_region_fidelity` further extends consistency checks to regions that **enter the field of view as the camera moves**, rather than only comparing persistently visible areas.

### Official versus Diagnostic Roles

On the current public benchmark surface, `novel_region_fidelity`, `reprojection_error`, `optical_flow_aepe`, and `met3r_consistency` are **official** when successful. `depth_accuracy`, `depth_collision`, `reconstruction_consistency`, and the GVGC-style epipolar metrics are currently reported separately from headline aggregation. This split reflects differences in supervision shape, failure modes, and suite eligibility. An empty MEt3R overlap mask is an explicit worst-case score (`0` normalized), while unreadable media and runtime errors remain failures excluded from the mean. Epipolar metrics depend on OpenCV feature matching and may return `not_applicable` in weak-texture or strongly non-rigid motion regimes, whereas reconstruction metrics are more sensitive to full-video decoding and third-party runtime availability.

### Interpretation Boundaries

`consistency_3d_4d` measures **internal geometric self-consistency of the generated video**, not instruction following, semantic correctness, or aesthetic quality. It does not replace camera/motion-control readings in [`action_control`](action_control.md), nor is it equivalent to style and perceptual scores in [`quality`](quality.md). $S_{\mathrm{3D/4D}}$ should be reported alongside action control, quality, and physics dimensions: a high score means the output is more easily explained by a unified 3D/4D structure, but it does not guarantee correct prompt execution or physically plausible object behavior.

</div>

</div>
