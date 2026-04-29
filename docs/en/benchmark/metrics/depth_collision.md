# Depth Collision

`depth_collision` uses Video-Depth-Anything metric depth to flag likely camera-path collisions. It is a geometry diagnostic, not a ground-truth physics contact metric.

## What It Measures

- whether the center of the camera view gets too close to foreground geometry
- rollout failures such as wall penetration, object collision, or face-close camera motion
- not semantic correctness, true depth accuracy, or physical contact correctness

## Current Algorithm

1. The metric requires a video prediction.
2. It reads sampled benchmark frames by default; `frame_source = full_video` reads the full prediction.
3. Video-Depth-Anything predicts metric depth for each frame.
4. The metric averages depth inside a centered crop, default `center_crop_ratio = 0.2`.
5. A collision is flagged when any centered-crop mean depth is below `depth_threshold`, default `1.0`.
6. `raw = 1.0` means collision detected; `raw = 0.0` means no collision detected.
7. Normalization is reversed: no collision maps to `1.0`, collision maps to `0.0`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled or full prediction video frames |
| Reference input | unused |
| Annotation input | unused |
| Backend | `video_depth_anything` |
| Default runtime | `thirdparty/Video-Depth-Anything`, `ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth`, `encoder = vits`, `input_size = 256` |
| Raw output | binary collision flag, lower is better |
| Normalized output | `[0, 1]`, higher is better |
| Detail fields | `center_crop_mean_depths`, `collision`, `first_collision_frame`, `threshold_results`, `depth_shape`, `frame_source`, `frame_indices` |

## Aggregation

| Item | Status |
| --- | --- |
| Suites | `image_static`, `image_dynamic`, `video_static`, `video_dynamic` |
| Aggregation | always `diagnostic` |
| Dimension | `consistency_3d_4d` |

Key limit: a high score only means the current center-crop depth threshold did not fire. It does not prove the camera path is physically valid.
