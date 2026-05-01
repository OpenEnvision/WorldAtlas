# 3D/4D Consistency

The `consistency_3d_4d` dimension covers geometry, multiview consistency, reprojection, and motion-field consistency. It asks whether the generated result supports a stable 3D/4D explanation, not just plausible individual frames.

## Scope

| Layer | Metrics |
| --- | --- |
| novel regions | [`novel_region_fidelity`](../novel_region_fidelity.md) |
| multiview / 3D representation consistency | [`met3r_consistency`](../met3r_consistency.md), [`reconstruction_consistency`](../reconstruction_consistency.md) |
| classical epipolar geometry | [`epipolar_error`](../gvgc_geometry.md#epipolar-error), [`epipolar_inlier_rate`](../gvgc_geometry.md#epipolar-inlier-rate), [`feature_match_coverage`](../gvgc_geometry.md#feature-match-coverage) |
| camera and reprojection | [`reprojection_error`](../reprojection_error.md) |
| depth | [`depth_accuracy`](../depth_accuracy.md), [`depth_collision`](../depth_collision.md) |
| motion field | [`optical_flow_aepe`](../optical_flow_aepe.md) |

## Reading Notes

- These metrics are closer to asking whether the geometry behind a generated video is self-consistent.
- `epipolar_error`, `epipolar_inlier_rate`, and `feature_match_coverage` are GVGC-style classical-geometry diagnostics. They use OpenCV feature matching and fundamental-matrix RANSAC, so they are most reliable for textured, mostly static scenes with camera motion.
- `reprojection_error` and `optical_flow_aepe` are usually official when successful; `depth_accuracy`, `depth_collision`, `met3r_consistency`, `reconstruction_consistency`, and the epipolar metrics are currently more diagnostic.
- This group does not directly measure prompt following or aesthetic quality, so it should be reported with `action_control` and `quality`.
