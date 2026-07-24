# Metrics Overview

This page follows the public benchmark surface declared in `config/benchmark.yaml`. The four formal suites are the public benchmark surface; `experimental` is a supplemental slice and should not be merged into a single headline benchmark total.

## Reading Rules

The report contract uses the following fields:

- **`raw`** — backend-native value; may be a scalar or a component dictionary
- **`normalized`** — `[0, 1]` score written into aggregation; higher is always better
- **`official`** — enters `per_metric_summary.json` and `per_dimension_summary.json`
- **`diagnostic`** — non-official; never enters headline aggregation
- **`not_applicable`** — strong supervision is absent, so no score is computed or aggregated; recorded in `not_applicable_summary.json`
- **`normalized = null`** — runtime failure or missing input; it is not automatically treated as zero

Per-metric normalization, suite coverage, and official eligibility are documented on each metric page.

## Suite Vectors

| Suite | Metrics | Notes |
| --- | ---: | --- |
| `image_static` | 44 | Image-conditioned static world generation; video predictions are evaluated with the video metric surface when inputs allow |
| `image_dynamic` | 34 | Same as `image_static`, usually with broader motion- and object-prompt coverage |
| `video_static` | 38 | Video-conditioned static world generation |
| `video_dynamic` | 44 | Video-conditioned dynamic world generation |
| `experimental` | 3 | Supplemental slice; not part of the formal headline benchmark |

Whether a metric is official, diagnostic, or `not_applicable` in a given suite depends on supervision and backend success; see each metric page and the [Benchmark Overview](../index.md).

## Six Evaluation Pillars

The Metrics navigation is organized by six top-level dimensions. Each dimension has its own index page with metric families and interpretation notes.

| Dimension | Index page |
| --- | --- |
| Long sequence (memory) | [Long Sequence (Memory)](dimensions/long_sequence.md) |
| Action control | [Action Control](dimensions/action_control.md) |
| 3D/4D consistency | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| Physics | [Physics](dimensions/physics.md) |
| Quality | [Quality](dimensions/quality.md) |
| Real time | [Real Time](dimensions/real_time.md) |

## Metric Index

Links below are grouped by dimension. Algorithm, backend, normalization, and aggregation status are documented on each metric page.

**Action control** — [`prompt_alignment`](prompt_alignment.md) · [`semantic_alignment`](semantic_alignment.md) · [`object_detection`](object_detection.md) · [`camera_error`](camera_error.md) · [`motion_accuracy`](motion_accuracy.md)

**Quality** — [`style_consistency`](style_consistency.md) · [`perceptual_quality`](perceptual_quality.md) · [`image_quality`](image_quality.md) · [`hpsv3_norm`](hpsv3_norm.md) · [`brightness_distribution_consistency`](brightness_consistency.md) · [`color_temperature_constraint`](color_temperature_consistency.md) · [`sharpness_retention`](sharpness_retention.md) · [`psnr`](psnr.md) · [`ssim`](ssim.md) · [`long_sequence_aesthetic_quality`](long_sequence.md) · [`long_sequence_imaging_quality`](long_sequence.md)

**Long sequence (memory)** — [`seen_region_preservation`](seen_region_preservation.md) · [`object_permanence`](object_permanence.md) · [`memory_revisit_consistency`](memory_revisit_consistency.md) · [`long_sequence_motion_smoothness`](long_sequence.md)

**3D/4D consistency** — [`novel_region_fidelity`](novel_region_fidelity.md) · [`reprojection_error`](reprojection_error.md) · [`optical_flow_aepe`](optical_flow_aepe.md) · [`epipolar_error`](gvgc_geometry.md#epipolar-error) · [`epipolar_inlier_rate`](gvgc_geometry.md#epipolar-inlier-rate) · [`feature_match_coverage`](gvgc_geometry.md#feature-match-coverage) · [`depth_accuracy`](depth_accuracy.md) · [`depth_collision`](depth_collision.md) · [`met3r_consistency`](met3r_consistency.md) · [`reconstruction_consistency`](reconstruction_consistency.md)

**Physics** — [`pmf`](pmf.md) · [`motion_magnitude`](motion_magnitude.md) · [`motion_smoothness`](motion_smoothness.md) · [`long_sequence_dynamic_degree`](long_sequence.md)

**Real time** — [`physical_fps_error`](temporal_calibration.md) · [`physical_fps_pct_error`](temporal_calibration.md) · [`physical_fps_intra_video_cv`](temporal_calibration.md) · [`effective_generation_fps`](realtime_throughput_latency.md#effective-generation-fps) · [`realtime_factor`](realtime_throughput_latency.md#realtime-factor) · [`action_response_latency`](realtime_throughput_latency.md#action-response-latency) · [`latency_budget_pass_rate`](realtime_throughput_latency.md#latency-budget-pass-rate)

## Separate Physics Track

`config/benchmark_physics.yaml` is the simulator-backed physics configuration, and it is not the same evaluation surface as the `experimental` suite in the default `config/benchmark.yaml`. It carries simulator-backed physics metrics and rolls them up into the top-level `physics` dimension. Metric semantics, artifact requirements, and comparability are governed by [Physics](dimensions/physics.md), [Simulator-Backed Physics](physics/index.md), and [Physics Metrics & Contracts](physics/metrics.md).
