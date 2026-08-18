# 3D/4D Consistency

The consistency_3d_4d dimension evaluates whether a generated rollout admits a **stable spatiotemporal geometric explanation** in three or four dimensions. Unlike quality metrics that judge single-frame plausibility or low-level appearance stability, this dimension asks whether camera motion, depth layering, cross-frame correspondence, newly exposed regions, and motion fields can be explained jointly under one geometric hypothesis across the full video.

**Dimension key:** consistency_3d_4d. Results roll up into per_dimension_summary.json.

**Metric families:** Six: novel regions, multiview, epipolar geometry, reprojection, depth, and motion.

**Core question:** Geometric coherence. Does the video evolve like a continuous viewpoint in a 3D world?

**Metric families and geometric semantics**

Readings are organized from visible regions to newly exposed regions, from local epipolar constraints to global reconstruction, and from depth structure to short-horizon motion fields.

**Novel regions.** Masked appearance fidelity when the reference camera path exposes regions unseen in the first frame. Metric: [novel_region_fidelity](../novel_region_fidelity.md).

**Multiview / 3D representation.** Consistency between anchor viewpoints and whether the prediction can be reconstructed as 3D Gaussians and re-rendered along the view path. Metrics: [met3r_consistency](../met3r_consistency.md) and [reconstruction_consistency](../reconstruction_consistency.md).

**Classical epipolar geometry.** Whether adjacent frames admit a stable fundamental-matrix / Sampson epipolar explanation. Metrics: [epipolar_error](../gvgc_geometry.md#epipolar-error), [epipolar_inlier_rate](../gvgc_geometry.md#epipolar-inlier-rate), and [feature_match_coverage](../gvgc_geometry.md#feature-match-coverage).

**Camera and reprojection.** Whether dense SLAM-recovered poses and depths reproject covisible points consistently onto the image plane. Metric: [reprojection_error](../reprojection_error.md).

**Depth structure.** Whether relative depth ordering is preserved and whether the forward viewing corridor shows implausible near geometry. Metrics: [depth_accuracy](../depth_accuracy.md) and [depth_collision](../depth_collision.md).

**Motion field.** Short-horizon photometric self-consistency under forward-backward optical-flow round-trips. Metric: [optical_flow_aepe](../optical_flow_aepe.md).

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

