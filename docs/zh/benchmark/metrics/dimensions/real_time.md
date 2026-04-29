# Real Time

`real_time` 维度覆盖节奏与 FPS 标定。它关注生成视频的物理时间节奏是否和 metadata / expected FPS 一致。

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| 平均物理 FPS 误差 | [`physical_fps_error`](../temporal_calibration.md) |
| 相对 FPS 误差 | [`physical_fps_pct_error`](../temporal_calibration.md) |
| 视频内部节奏稳定性 | [`physical_fps_intra_video_cv`](../temporal_calibration.md) |

## 解读重点

- 这些指标由 [`temporal_calibration`](../temporal_calibration.md) 页面统一说明。
- 它们回答“视频播放速度/物理节奏是否校准”，不直接评价视觉质量或语义正确性。
- 当模型输出 FPS metadata 或 motion cadence 与 reference 不一致时，这组指标会暴露时间标定问题。
