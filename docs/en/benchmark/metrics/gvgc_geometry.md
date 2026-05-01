# Epipolar Geometry Diagnostics

WorldArena uses this page for the three GVGC-style classical-geometry metrics now grouped under `consistency_3d_4d`:

| Metric | Direction | Raw value |
| --- | --- | --- |
| `epipolar_error` | lower is better | median inlier Sampson error after fundamental-matrix RANSAC |
| `epipolar_inlier_rate` | higher is better | RANSAC inliers divided by Lowe-ratio feature matches |
| `feature_match_coverage` | higher is better | RANSAC inliers divided by the smaller keypoint count in the frame pair |

These metrics are lightweight geometric consistency probes. They do not replace DROID-SLAM reprojection, MEt3R, or reconstruction-based scores; instead, they provide a transparent feature-geometry signal that is cheap to run and easy to inspect.

## What They Measure

The core question is whether nearby generated frames admit a stable two-view epipolar geometry. If the same static scene points appear across two frames, their image locations should be explainable by a single fundamental matrix \(F\). Strong videos should therefore produce many geometrically consistent matches with low epipolar residuals.

This is useful for world-model evaluation because 3D/4D inconsistency often appears as broken correspondences: backgrounds slide, rigid surfaces bend, duplicated structures drift independently, or camera motion cannot be explained by a single coherent scene geometry.

## Algorithm

1. The metric receives the loaded prediction video frames.
2. Each frame is converted to grayscale and resized so the analysis width is at most `max_analysis_width`.
3. Frame pairs are selected with `pair_stride`; at most `max_pairs` pairs are evaluated.
4. The backend detects local features with SIFT by default. If SIFT is unavailable, or if `feature_detector: orb` is configured, it uses ORB.
5. Descriptors are matched with a brute-force matcher and filtered by Lowe's ratio test.
6. If a pair has fewer than `min_matches`, that pair is marked failed.
7. The backend estimates a fundamental matrix with OpenCV RANSAC, preferring `USAC_MAGSAC` when available.
8. RANSAC inliers are scored with Sampson distance.
9. Per-pair raw values are aggregated across valid pairs:
    - `epipolar_error`: median of per-pair median inlier errors
    - `epipolar_inlier_rate`: mean of per-pair inlier rates
    - `feature_match_coverage`: mean of per-pair coverage values

If no pair has enough valid inliers, all three metrics return `raw = null`, `normalized = null`, and an error string.

## Metric Definitions

### `epipolar_error` {#epipolar-error}

For a matched point pair \(x_i, x'_i\) and the estimated fundamental matrix \(F\), the backend computes the Sampson error:

\[
d_i =
\frac{|{x'_i}^{T} F x_i|}
{\sqrt{(F x_i)_1^2 + (F x_i)_2^2 + (F^T x'_i)_1^2 + (F^T x'_i)_2^2}}
\]

For each valid frame pair, `epipolar_error_pair` is the median \(d_i\) over RANSAC inliers. The video-level raw value is the median across valid frame pairs.

Normalization uses the configured range `[0, 5]` with lower-is-better scaling:

\[
score = 1 - \mathrm{clamp}\left(\frac{raw - 0}{5 - 0}, 0, 1\right)
\]

Interpretation: low values mean the inlier correspondences fit a coherent epipolar geometry. High values suggest geometric drift, weak two-view consistency, or unstable feature matches.

### `epipolar_inlier_rate` {#epipolar-inlier-rate}

For each valid frame pair:

\[
rate =
\frac{\#\mathrm{RANSAC\ inliers}}{\#\mathrm{good\ feature\ matches}}
\]

The video-level raw value is the mean rate across valid frame pairs. Normalization is identity over `[0, 1]`.

Interpretation: high values mean that most descriptor matches can be explained by one fundamental matrix. Low values indicate many contradictory correspondences, which often happens with inconsistent geometry, large non-rigid changes, or severe generation artifacts.

### `feature_match_coverage` {#feature-match-coverage}

For each valid frame pair:

\[
coverage =
\min\left(
\frac{\#\mathrm{RANSAC\ inliers}}
{\min(\#\mathrm{keypoints}_{left}, \#\mathrm{keypoints}_{right})},
1
\right)
\]

The video-level raw value is the mean coverage across valid frame pairs. Normalization is identity over `[0, 1]`.

Interpretation: this asks whether the epipolar-consistent matches cover a meaningful portion of the detectable image structure. It helps distinguish a video with many stable correspondences from one where RANSAC succeeds only on a tiny textured patch.

## Configuration

Default runtime settings live under `benchmark.metric_runtime.gvgc_geometry`:

| Key | Default | Meaning |
| --- | ---: | --- |
| `feature_detector` | `sift` | detector choice; `orb` is also supported |
| `max_features` | `2000` | maximum local features per frame |
| `ratio_test` | `0.75` | Lowe-ratio match filtering threshold |
| `min_matches` | `12` | minimum good matches before attempting RANSAC |
| `max_analysis_width` | `640` | downsample width cap for geometry analysis |
| `ransac_reproj_threshold` | `1.0` | OpenCV fundamental-matrix RANSAC threshold |
| `ransac_confidence` | `0.999` | RANSAC confidence |
| `ransac_max_iters` | `10000` | RANSAC iteration cap |
| `pair_stride` | `1` | frame gap between evaluated pairs |
| `max_pairs` | `8` | maximum number of frame pairs to score |

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | loaded prediction video frames |
| Reference input | none |
| Annotation input | none |
| Backend | `opencv_epipolar` |
| Aggregation | `diagnostic` in all formal suites |
| Details payload | frame count, pair count, valid/failed pair counts, detector, runtime settings, and per-pair match/inlier/error statistics |
| Failure mode | `raw = null`, `normalized = null`, plus an error when the prediction is not video or no valid epipolar pair is found |

## Reading Notes

- These metrics are most reliable on textured, mostly static scenes with camera motion.
- They are less reliable on textureless rooms, repeated patterns, heavy motion blur, large dynamic foregrounds, or cuts between unrelated views.
- A good `epipolar_error` alone is not enough: a pair may fit a simple geometry while covering only a small region, so read it together with `epipolar_inlier_rate` and `feature_match_coverage`.
- High coverage with high error usually means the video has many correspondences but poor geometric agreement.
- Low coverage with low error usually means RANSAC found a small clean subset, but the evidence is too narrow for a strong geometry claim.
