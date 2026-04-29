# Real Time

The `real_time` dimension covers rhythm and FPS calibration. It asks whether the physical tempo of the generated video agrees with metadata or expected FPS.

## Scope

| Layer | Metrics |
| --- | --- |
| average physical FPS error | [`physical_fps_error`](../temporal_calibration.md) |
| relative FPS error | [`physical_fps_pct_error`](../temporal_calibration.md) |
| intra-video rhythm stability | [`physical_fps_intra_video_cv`](../temporal_calibration.md) |

## Reading Notes

- These metrics are documented together in [`temporal_calibration`](../temporal_calibration.md).
- They answer whether playback speed or physical tempo is calibrated, not whether visual quality or semantics are correct.
- If output FPS metadata or motion cadence diverges from the reference, this group exposes the timing-calibration issue.
