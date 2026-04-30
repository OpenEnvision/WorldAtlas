# Metrics Overview

This page follows the public benchmark surface declared in `config/benchmark.yaml`. The four formal suites are the public benchmark surface; `experimental` is a supplemental slice and should not be merged into a single headline benchmark total.

## Reading Rules

| Field | Meaning |
| --- | --- |
| `raw` | backend-native value; may be a scalar or a component dictionary |
| `normalized` | `[0, 1]` score written into aggregation; higher is always better |
| `official` | enters `per_metric_summary.json` and `per_dimension_summary.json` |
| `diagnostic` | diagnostics only; never enters headline aggregation |
| `not_applicable` | strong supervision is absent, so no score is computed or aggregated; recorded in `not_applicable_summary.json` |
| `normalized = null` | runtime failure or missing input; it is not automatically treated as zero |

## Suite Vectors

| Suite | Metrics | Official / conditionally official | Diagnostic / not applicable |
| --- | ---: | --- | --- |
| `image_static` | 31 | image-to-video predictions are evaluated with the video-output metric superset where inputs allow it. `style_consistency`, `perceptual_quality`, `reprojection_error`, `optical_flow_aepe`, `motion_magnitude`, `motion_smoothness`, and `physical_fps_*` are official when they produce numeric scores. `prompt_alignment`, `object_detection`, `camera_error`, and `motion_accuracy` are gated by supervision. `seen_region_preservation`, `novel_region_fidelity`, `object_permanence`, and `pmf` require compatible reference video or masks and are often empty for pure image references. | `semantic_alignment`, `brightness_consistency`, `color_temperature_consistency`, `sharpness_retention`, `depth_accuracy`, `depth_collision`, `psnr`, `ssim`, `met3r_consistency`, `reconstruction_consistency`, `long_sequence_motion_smoothness`, `long_sequence_dynamic_degree`, `long_sequence_aesthetic_quality`, `long_sequence_imaging_quality` |
| `image_dynamic` | 31 | same as `image_static`, usually with broader motion-prompt and object-prompt coverage | same as `image_static` |
| `video_static` | 19 | `style_consistency`, `perceptual_quality`, `seen_region_preservation`, `novel_region_fidelity`, `reprojection_error`, and `optical_flow_aepe`; `prompt_alignment`, `object_detection`, and `camera_error` are supervision-gated | `semantic_alignment`, `brightness_consistency`, `color_temperature_consistency`, `sharpness_retention`, `depth_accuracy`, `depth_collision`, `psnr`, `ssim`, `met3r_consistency`, `reconstruction_consistency` |
| `video_dynamic` | 25 | `style_consistency`, `perceptual_quality`, `object_permanence`, `pmf`, `motion_magnitude`, `motion_smoothness`, and `physical_fps_*`; `prompt_alignment` and `motion_accuracy` are supervision-gated | `semantic_alignment`, `brightness_consistency`, `color_temperature_consistency`, `sharpness_retention`, `depth_accuracy`, `depth_collision`, `psnr`, `ssim`, `met3r_consistency`, `reconstruction_consistency`, `long_sequence_motion_smoothness`, `long_sequence_dynamic_degree`, `long_sequence_aesthetic_quality`, `long_sequence_imaging_quality` |
| `experimental` | 3 | in the current implementation, `motion_magnitude` and `motion_smoothness` become official when they produce numeric scores; this suite is not part of the formal headline benchmark | `memory_revisit_consistency` |

## Six Evaluation Pillars

The Metrics navigation is organized by these six top-level dimensions, which also appear in `per_dimension_summary.json`.

| Key | Label | Practical meaning | Dimension page |
| --- | --- | --- | --- |
| `long_sequence` | Long sequence (memory) | state memory, appearance preservation, and long-rollout stability | [Long Sequence (Memory)](dimensions/long_sequence.md) |
| `action_control` | Action control | scene control, camera control, and object/action control | [Action Control](dimensions/action_control.md) |
| `consistency_3d_4d` | 3D/4D consistency | geometry, multiview, reprojection, and motion-field consistency | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| `physics` | Physics | physical motion realism and simulator-backed physics metrics | [Physics](dimensions/physics.md) |
| `quality` | Quality | style, perceptual quality, low-level appearance stability, and reconstruction-style fidelity | [Quality](dimensions/quality.md) |
| `real_time` | Real time | tempo and FPS calibration | [Real Time](dimensions/real_time.md) |

## Complete Metric Inventory

| Metric | Dimension | Backend | Suites | Raw value and normalization | Aggregation |
| --- | --- | --- | --- | --- | --- |
| [`prompt_alignment`](prompt_alignment.md) | `action_control` | `reference` | formal 4 suites | CLIPScore-like `[0, 1]` scalar on video-prediction anchor frames; identity scaling | official only with explicit instruction supervision and non-self-conditioned inputs; non-video predictions are submission errors |
| [`semantic_alignment`](semantic_alignment.md) | `action_control` | `clip` | formal 4 suites | CLIP similarity between prediction frames and reference caption/instruction | diagnostic |
| [`object_detection`](object_detection.md) | `action_control` | `grounding_dino` | image 2 suites, `video_static` | object-prompt detection success rate in `[0, 1]` | official with explicit object annotations; caption-derived objects are `not_applicable` |
| [`camera_error`](camera_error.md) | `action_control` | `droid_slam` | `image_static`, `image_dynamic`, `video_static` | `{rotation_error, translation_error}` reduced by geometric mean; lower is better | official when pose annotations exist and the backend succeeds |
| [`motion_accuracy`](motion_accuracy.md) | `action_control` | `sam2_sea_raft` | `image_static`, `image_dynamic`, `video_dynamic` | mask-local motion accuracy mapped by z-score | official when dynamic masks exist and the backend succeeds |
| [`style_consistency`](style_consistency.md) | `quality` | `reference` | formal 4 suites | VGG Gram style similarity in `[0, 1]` | official |
| [`perceptual_quality`](perceptual_quality.md) | `quality` | `reference` | formal 4 suites | reference IQA backend scalar in `[0, 1]` | official |
| [`brightness_consistency`](brightness_consistency.md) | `quality` | `luminance_temporal` | formal 4 suites | luminance temporal-stability proxy in `[0, 1]` | diagnostic |
| [`color_temperature_consistency`](color_temperature_consistency.md) | `quality` | `hsv_hue_drift` | formal 4 suites | HSV hue-drift stability proxy in `[0, 1]` | diagnostic |
| [`sharpness_retention`](sharpness_retention.md) | `quality` | `tenengrad_brisque` | formal 4 suites | Tenengrad retention gated by BRISQUE in `[0, 1]` | diagnostic |
| [`psnr`](psnr.md) | `quality` | `framewise` | formal 4 suites | paired-frame PSNR; forward scaling over `[10, 40]` | diagnostic |
| [`ssim`](ssim.md) | `quality` | `framewise` | formal 4 suites | paired-frame SSIM in `[0, 1]` | diagnostic |
| [`long_sequence_aesthetic_quality`](long_sequence.md) | `quality` | `long_sequence` | `image_static`, `image_dynamic`, `video_dynamic` | Long-Sequence aesthetic quality in `[0, 1]` | diagnostic |
| [`long_sequence_imaging_quality`](long_sequence.md) | `quality` | `long_sequence` | `image_static`, `image_dynamic`, `video_dynamic` | Long-Sequence imaging quality in `[0, 1]` | diagnostic |
| [`seen_region_preservation`](seen_region_preservation.md) | `long_sequence` | `flow_dino` | `image_static`, `image_dynamic`, `video_static` | PSNR/DINO composite inside first-frame seen mask | official when successful; pure image references are usually empty |
| [`object_permanence`](object_permanence.md) | `long_sequence` | `mask_dino` | `image_static`, `image_dynamic`, `video_dynamic` | dynamic-mask identity-curve match between reference and prediction | official with compatible video reference and mask |
| [`memory_revisit_consistency`](memory_revisit_consistency.md) | `long_sequence` | `dinov2` | `experimental` | loop-closure similarity times mid-rollout progress gate in `[0, 1]` | diagnostic |
| [`long_sequence_motion_smoothness`](long_sequence.md) | `long_sequence` | `long_sequence` | `image_static`, `image_dynamic`, `video_dynamic` | Long-Sequence motion smoothness in `[0, 1]` | diagnostic |
| [`novel_region_fidelity`](novel_region_fidelity.md) | `consistency_3d_4d` | `flow_dino` | `image_static`, `image_dynamic`, `video_static` | PSNR/DINO composite inside novel mask | official when successful; pure image references are usually empty |
| [`reprojection_error`](reprojection_error.md) | `consistency_3d_4d` | `droid_slam` | `image_static`, `image_dynamic`, `video_static` | DROID reprojection error; reverse scaling over `[0, 1.0719]` | official when successful |
| [`optical_flow_aepe`](optical_flow_aepe.md) | `consistency_3d_4d` | `sea_raft` | `image_static`, `image_dynamic`, `video_static` | prediction optical-flow AEPE backend score; reverse scaling over `[0, 1.192]` | official when successful |
| [`depth_accuracy`](depth_accuracy.md) | `consistency_3d_4d` | `depth_anything` | formal 4 suites | paired-frame AbsRel; reverse scaling over `[0, 1]` | diagnostic |
| [`depth_collision`](depth_collision.md) | `consistency_3d_4d` | `video_depth_anything` | formal 4 suites | center-region near-depth collision flag `0/1`; reverse scaling | diagnostic |
| [`met3r_consistency`](met3r_consistency.md) | `consistency_3d_4d` | `met3r` | formal 4 suites | paired-anchor MEt3R distance; reverse scaling over `[0, 2]` | diagnostic |
| [`reconstruction_consistency`](reconstruction_consistency.md) | `consistency_3d_4d` | reconstruction LPIPS backend | formal 4 suites | `S_recon = clip(1 - mean LPIPS, 0, 1)` | diagnostic |
| [`pmf`](pmf.md) | `physics` | `numpy_fft` | `image_static`, `image_dynamic`, `video_dynamic` | prediction/reference 3D FFT energy-distribution similarity; scaling over `[0, 6]` | official when successful; requires at least two reference frames |
| [`motion_magnitude`](motion_magnitude.md) | `physics` | `sea_raft` | `image_static`, `image_dynamic`, `video_dynamic`, `experimental` | optical-flow magnitude mapped by z-score into `[0.25, 0.75]` | official when successful |
| [`motion_smoothness`](motion_smoothness.md) | `physics` | `vfi_mamba` | `image_static`, `image_dynamic`, `video_dynamic`, `experimental` | normalized `{MSE, SSIM, LPIPS}` components reduced by arithmetic mean | official when successful |
| [`long_sequence_dynamic_degree`](long_sequence.md) | `physics` | `long_sequence` | `image_static`, `image_dynamic`, `video_dynamic` | Long-Sequence dynamic degree in `[0, 1]` | diagnostic |
| [`physical_fps_error`](temporal_calibration.md) | `real_time` | `pulse_of_motion` | `image_static`, `image_dynamic`, `video_dynamic` | `abs(avg_phyfps - meta_fps)`; reverse scaling over `[0, 30]` | official when successful |
| [`physical_fps_pct_error`](temporal_calibration.md) | `real_time` | `pulse_of_motion` | `image_static`, `image_dynamic`, `video_dynamic` | relative FPS error; reverse scaling over `[0, 1]` | official when successful |
| [`physical_fps_intra_video_cv`](temporal_calibration.md) | `real_time` | `pulse_of_motion` | `image_static`, `image_dynamic`, `video_dynamic` | intra-video physical-FPS coefficient of variation; reverse scaling over `[0, 1]` | official when successful |

## Separate Physics Track

`config/benchmark_physics.yaml` is the simulator-backed physics configuration, and it is not the same evaluation surface as the `experimental` suite in the default `config/benchmark.yaml`. It carries `mse`, `st_iou`, `depth_warp_charb`, `rgb_warp_charb`, `chamfer_4d`, `worldline_*`, `trajectory_*`, `final_state_accuracy`, and related metrics, and rolls them up into the top-level `physics` dimension. It uses the same report contract, but metric semantics, artifact requirements, and comparability are governed by [Physics](dimensions/physics.md), [Simulator-Backed Physics](physics/index.md), and [Physics Metrics & Contracts](physics/metrics.md).
