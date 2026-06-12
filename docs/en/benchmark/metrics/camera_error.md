# Camera Error

`camera_error` measures how closely the camera trajectory recovered from the prediction matches the annotated camera poses.

## What It Measures

- camera-motion fidelity relative to pose annotations
- viewpoint control, not object motion control
- agreement with annotated extrinsics after post-alignment

## Detailed Algorithm

1. WorldArena decodes the prediction video into frame PNGs and counts all decoded frames.
2. `load_camera_matrices()` loads `poses.npy` from the sample annotation directory. If the stored poses are not already aligned with the decoded frame count, they are resampled by:
    - linear interpolation for translation
    - quaternion slerp for rotation
    - optional `indexes.txt` positions when available
3. The wrapped third-party metric runs `DROID-SLAM` on the decoded frames to recover a camera trajectory from the prediction.
4. The third-party implementation then converts the recovered trajectory into the coordinate convention expected by its evaluation code.
5. Translation is scale-aligned to the ground truth by solving a scalar optimization problem `min_c sum ||t_gt - c * t_pred||`. If that optimization fails, the code falls back to a simple mean-distance ratio.
6. Rotation error is the mean relative rotation angle in degrees after alignment.
7. Translation error is the mean Euclidean distance after scalar alignment.
8. WorldArena normalizes the two components with:
    - rotation error range `[0.0, 15.0]`, lower is better
    - translation error range `[0.0, 0.5]`, lower is better
9. The final normalized score is the geometric mean of the two normalized components.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | `poses.npy`; optional `indexes.txt` for pose resampling |
| Raw output | `{"rotation_error": float, "translation_error": float}` |
| Normalized output | one scalar in `[0, 1]` after component normalization and geometric mean reduction |
| Details payload | `frame_count`, `camera_scale`, `native_components` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if pose data or the DROID runtime is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when pose annotations exist and a numeric score is produced |
| Metric role | `core camera-control metric` |
| Publication verdict | usable as a core camera-controllability metric, with the fixed-intrinsics and scale-invariant alignment assumptions stated |

- The wrapped DROID-SLAM path follows the upstream setup and uses a **fixed synthetic intrinsics vector** (`[500, 500, 256, 256]`, scaled with frame size) inside the third-party metric.
- This means the metric should be interpreted as an internally consistent camera controllability error under a fixed-intrinsics assumption, not as an absolute per-sample calibrated camera reconstruction error.
- The translation component is post-aligned by a least-squares scalar, so the metric emphasizes camera-motion direction, rotation control, and scale-invariant trajectory agreement rather than strict metric-space scale fidelity.
- Recommended reporting: use it as a core camera-control metric, while stating that it is DROID-SLAM-based and uses fixed synthetic intrinsics with scale-invariant alignment.
