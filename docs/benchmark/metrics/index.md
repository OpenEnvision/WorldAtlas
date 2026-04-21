# Metrics Overview

This section explains metric semantics and reporting rules. Dataset coverage and collection statistics stay on [Dataset Analytics](../../overview.md).

## Metric Families

| Dimension | Metrics | Typical suites | Notes |
| --- | --- | --- | --- |
| content alignment | [`prompt_alignment`](prompt_alignment.md) | all official suites | official only with strong instruction supervision |
| style consistency | [`style_consistency`](style_consistency.md) | all official suites | reference-conditioned appearance similarity |
| perceptual quality | [`perceptual_quality`](perceptual_quality.md) | all official suites | no reference required |
| appearance alignment | [`reference_appearance_alignment`](reference_appearance_alignment.md), [`anchor_consistency`](anchor_consistency.md) | `auxiliary`, `experimental` | diagnostic by design |
| camera control | [`camera_error`](camera_error.md) | `video_static` | needs pose annotations |
| object control | [`object_detection`](object_detection.md) | `video_static` | official only with explicit object annotations |
| geometry consistency | [`reprojection_error`](reprojection_error.md) | `video_static` | prediction-only geometric self-consistency |
| photometric consistency | [`optical_flow_aepe`](optical_flow_aepe.md) | `video_static` | prediction-only flow error |
| motion control | [`motion_accuracy`](motion_accuracy.md) | `video_dynamic` | needs masks |
| motion dynamics | [`motion_magnitude`](motion_magnitude.md), [`motion_smoothness`](motion_smoothness.md) | `video_dynamic`, `experimental` | amplitude and temporal smoothness are separate signals |

## How To Read A Metric Page

Each metric page answers five questions:

1. What signal the metric is trying to capture.
2. Which inputs it actually uses.
3. What the raw output looks like.
4. How it is normalized into `[0, 1]`.
5. When the score is official, weakly supervised, or diagnostic.

Do not assume two metrics with the same `[0, 1]` scale are interchangeable. Their raw signals, supervision assumptions, and normalization schemes are different.
