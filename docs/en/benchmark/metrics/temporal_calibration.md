# Temporal Calibration

The temporal-calibration family belongs to WorldArena's top-level `real_time` dimension. It asks a different question from optical-flow or smoothness metrics: does the generated video move at a physically plausible rate relative to its declared FPS?

WorldArena currently computes this family with the wrapped `Pulse-of-Motion` or `Visual Chronometer` predictor.

## What It Measures

- `physical_fps_error`: absolute mismatch between estimated physical FPS and the video's FPS metadata
- `physical_fps_pct_error`: the same mismatch divided by the metadata FPS
- `physical_fps_intra_video_cv`: coefficient of variation across per-segment physical FPS estimates

These metrics measure tempo plausibility. They do **not** measure prompt following, object control, or camera accuracy.

## Detailed Algorithm

1. The runtime reads the full prediction video.
2. It extracts overlapping segments with the current runtime defaults:
    - `clip_length = 30`
    - `stride = 4`
    - `resolution = 216`
3. Each segment is resized to `216 x 216`, normalized into `[-1, 1]`, and arranged as `[C, T, H, W]`.
4. The wrapped `Pulse-of-Motion` model predicts a log physical FPS for each segment.
5. The predictor exponentiates that value to obtain `predicted_phyfps` per segment.
6. The runtime averages all segment predictions to get `avg_phyfps`.
7. It obtains `meta_fps` from:
    - the prediction container FPS when OpenCV can read it
    - otherwise `sample.fps` from the benchmark manifest
8. It then computes:
    - `physical_fps_error = abs(avg_phyfps - meta_fps)`
    - `physical_fps_pct_error = physical_fps_error / meta_fps`
    - `physical_fps_intra_video_cv = std(segment_phyfps) / mean(segment_phyfps)`
9. WorldArena normalizes each metric with direct bounded scaling:
    - `physical_fps_error`: `[0, 30]`, lower is better
    - `physical_fps_pct_error`: `[0, 1]`, lower is better
    - `physical_fps_intra_video_cv`: `[0, 1]`, lower is better

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full prediction video |
| Reference input | none |
| Annotation input | fallback manifest FPS only if container FPS is missing |
| Raw output | one scalar per metric family member |
| Normalized output | one scalar in `[0, 1]`, where higher is better after inversion |
| Details payload | `avg_phyfps`, `segment_phyfps`, `segment_count`, `total_frames`, `meta_fps`, `meta_fps_source`, `device`, `clip_length`, `stride`, `resolution` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the side runtime, checkpoint, or FPS metadata is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | all three metrics are currently `official` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- These metrics are useful because they test a failure mode that ordinary flow and smoothness metrics miss: implausible physical tempo.
- They are still secondary metrics because the signal depends on an external pretrained tempo estimator and on trustworthy FPS metadata.
- Re-encoding a prediction with different container FPS can change the score even if the frame content stays the same.
- Recommended use: report these metrics when temporal calibration matters, but do not treat them as substitutes for motion-control or overall quality metrics.
- In the broader `real_time` pillar, these metrics are only the current public instantiation. Future online or interactive tracks may add latency, throughput, and wall-clock responsiveness under the same pillar.
