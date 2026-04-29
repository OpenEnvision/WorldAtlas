# Metrics Overview

This section maps the metric surface declared in `config/benchmark.yaml`. Each metric page explains the runtime inputs, backend, score shape, and interpretation details for one metric or metric family.

Dataset coverage and collection statistics stay on [Dataset Analytics](../../overview.md).

## Status Labels

- `official`: enters `per_metric_summary.json` when a numeric score is produced.
- `diagnostic`: never enters official aggregation.
- `not_applicable`: strong supervision is absent; no score is computed or aggregated.
- `raw = null`, `normalized = null`: the metric could not produce a usable score for that sample.

## Six Evaluation Pillars

`per_dimension_summary.json` uses six top-level keys, and the Metrics navigation is now organized by the same six dimensions:

| Key | Label | Practical meaning | Dimension page |
| --- | --- | --- | --- |
| `long_sequence` | Long sequence (memory) | state memory, appearance preservation, and long-rollout stability | [Long Sequence (Memory)](dimensions/long_sequence.md) |
| `action_control` | Action control | scene control, camera control, and object/action control | [Action Control](dimensions/action_control.md) |
| `consistency_3d_4d` | 3D/4D consistency | geometry, multiview, reprojection, and motion-field consistency | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| `physics` | Physics | physical motion realism and simulator-backed physics metrics | [Physics](dimensions/physics.md) |
| `quality` | Quality | style, perceptual quality, and reconstruction-style fidelity | [Quality](dimensions/quality.md) |
| `real_time` | Real time | tempo and FPS calibration | [Real Time](dimensions/real_time.md) |

Action control is intentionally broader than today's text-only prompt setting:

- camera control includes text instruction and interactive camera input such as keyboard or mouse control
- object control includes text instruction plus action streams such as keyboard input
- scene control focuses on text instruction over scene layout and semantic targets

The current public benchmark mostly instantiates this dimension through text prompts, structured object annotations, masks, pose metadata, and rollout prefixes. Keyboard and mouse traces belong to the same taxonomy, but are more naturally exercised in interactive or simulator-backed tracks.

This taxonomy is intentionally coarse-grained. The six pillars describe benchmark-level capability coverage; they do not imply that the current public benchmark already exhausts every control mode or every future metric family.

`image_static` and `image_dynamic` are image-conditioned video-generation suites. Their predictions are parsed as videos, so they now run the video-output metric superset where inputs permit. Prediction-only video metrics score normally; metrics that require reference video, pose, or masks return unavailable when those sidecars are absent.

## Current Metric Map

| Top-level dimension | Representative sub-problems | Representative metrics today | Typical suites | Current behavior |
| --- | --- | --- | --- | --- |
| Long sequence (memory) | scene and appearance memory | [`seen_region_preservation`](seen_region_preservation.md), [`anchor_consistency`](anchor_consistency.md), [`subject_consistency`](subject_consistency.md), [`background_consistency`](background_consistency.md), [`object_permanence`](object_permanence.md), [`long_sequence_subject_consistency`](long_sequence.md), [`long_sequence_background_consistency`](long_sequence.md), [`long_sequence_motion_smoothness`](long_sequence.md) | all video-output formal suites, `auxiliary`, `experimental` | mix of official, unavailable, not-applicable, and diagnostic signals depending on side inputs |
| Action control | scene control (text instruction) | [`prompt_alignment`](prompt_alignment.md), [`semantic_alignment`](semantic_alignment.md) | all formal suites | `prompt_alignment` is computed on video-prediction anchor frames and is official only with explicit instruction supervision; `semantic_alignment` is diagnostic |
| Action control | camera control (text instruction, keyboard/mouse input) | [`camera_error`](camera_error.md) | `video_static` | `official` when pose annotations exist |
| Action control | object control (text instruction, keyboard input) | [`object_detection`](object_detection.md), [`motion_accuracy`](motion_accuracy.md) | image suites, `video_static`, `video_dynamic` | `object_detection` is official only with explicit object annotations; `motion_accuracy` is official when masks exist |
| 3D/4D consistency | geometry, multiview, and motion-field consistency | [`novel_region_fidelity`](novel_region_fidelity.md), [`reprojection_error`](reprojection_error.md), [`depth_accuracy`](depth_accuracy.md), [`depth_collision`](depth_collision.md), [`optical_flow_aepe`](optical_flow_aepe.md), [`met3r_consistency`](met3r_consistency.md), [`reconstruction_consistency`](reconstruction_consistency.md) | all formal suites | `reprojection_error` and `optical_flow_aepe` are official when numeric; reference-video metrics can be unavailable on pure image references; `reconstruction_consistency` is diagnostic |
| Physics | motion realism in the public benchmark | [`pmf`](pmf.md), [`motion_magnitude`](motion_magnitude.md), [`motion_smoothness`](motion_smoothness.md), [`long_sequence_dynamic_degree`](long_sequence.md) | image-to-video suites, `video_dynamic`, `experimental` | prediction-only motion metrics are official when numeric; `pmf` needs a multi-frame reference |
| Quality | appearance, perceptual quality, and reconstruction fidelity | [`style_consistency`](style_consistency.md), [`perceptual_quality`](perceptual_quality.md), [`psnr`](psnr.md), [`ssim`](ssim.md), [`long_sequence_aesthetic_quality`](long_sequence.md), [`long_sequence_imaging_quality`](long_sequence.md) | all formal suites, `video_dynamic` | `style_consistency` and `perceptual_quality` are official; the rest are diagnostic |
| Real time | FPS and tempo calibration | [`physical_fps_error`](temporal_calibration.md), [`physical_fps_pct_error`](temporal_calibration.md), [`physical_fps_intra_video_cv`](temporal_calibration.md) | `video_dynamic` | official when the side runtime and FPS metadata are available |

Read this table as the current public instantiation, not as a closed ontology. Future interactive tracks can add richer keyboard or mouse control, online latency, throughput, and denser world-state metrics while staying inside the same six pillars.

## Separate Physics-Only Metrics

`config/benchmark_physics.yaml` defines a simulator-backed physics config that also uses the suite label `experimental`. It carries metrics such as `mse`, `st_iou`, `depth_warp_charb`, `rgb_warp_charb`, `chamfer_4d`, and `worldline_*`, and all of them roll up into the top-level `physics` dimension for that physics-only track.

See [Physics](dimensions/physics.md) and [Simulator-Backed Physics](physics/index.md) for scope and reporting behavior, and [Physics Metrics & Contracts](physics/metrics.md) for the grouped metric contract, normalization ranges, and artifact requirements.

## How To Read A Metric Page

Each metric page answers the same practical questions:

1. what the metric is trying to capture
2. which inputs it actually uses
3. what the raw output looks like
4. how it is normalized into `[0, 1]`
5. when the score is `official`, `diagnostic`, or `not_applicable`
