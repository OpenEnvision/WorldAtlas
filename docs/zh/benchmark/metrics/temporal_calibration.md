# Temporal Calibration

`temporal_calibration` 这一组指标属于 WorldArena 顶层 `real_time` 维度。它问的是一个和 optical-flow、smoothness 都不同的问题：生成视频的运动节奏，是否与它声明的 FPS 相匹配、并且在物理上合理。

WorldArena 当前通过被包装的 `Pulse-of-Motion` 或 `Visual Chronometer` 预测器来计算这组指标。

## 这个指标测什么

- `physical_fps_error`：估计物理 FPS 与视频 FPS 元数据之间的绝对偏差
- `physical_fps_pct_error`：同样的偏差，只是再除以元数据 FPS
- `physical_fps_intra_video_cv`：单个视频内部各片段物理 FPS 预测的变异系数

这组指标测的是节奏合理性，而不是 prompt following、object control 或 camera accuracy。

## 具体算法

1. runtime 读取完整的预测视频。
2. 它会按当前默认参数提取重叠片段：
    - `clip_length = 30`
    - `stride = 4`
    - `resolution = 216`
3. 每个片段都会被 resize 到 `216 x 216`，归一化到 `[-1, 1]`，并排成 `[C, T, H, W]`。
4. 被包装的 `Pulse-of-Motion` 模型会对每个片段预测一个 log physical FPS。
5. 这个值会被取指数，得到该片段的 `predicted_phyfps`。
6. runtime 对全部片段预测取平均，得到 `avg_phyfps`。
7. `meta_fps` 的来源顺序是：
    - 如果 OpenCV 能读到预测视频的 container FPS，就用它
    - 否则退回到 benchmark manifest 里的 `sample.fps`
8. 然后计算三项指标：
    - `physical_fps_error = abs(avg_phyfps - meta_fps)`
    - `physical_fps_pct_error = physical_fps_error / meta_fps`
    - `physical_fps_intra_video_cv = std(segment_phyfps) / mean(segment_phyfps)`
9. WorldArena 对这三个指标都使用直接的有界缩放：
    - `physical_fps_error`：`[0, 30]`，越小越好
    - `physical_fps_pct_error`：`[0, 1]`，越小越好
    - `physical_fps_intra_video_cv`：`[0, 1]`，越小越好

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整预测视频 |
| 参考输入 | 无 |
| 标注输入 | 只有在 container FPS 缺失时，才退回使用 manifest 里的 FPS |
| 原始输出 | 每个 family 成员各自输出一个标量 |
| 归一化输出 | 反向缩放后的单个 `[0, 1]` 标量，越高越好 |
| 细节字段 | `avg_phyfps`、`segment_phyfps`、`segment_count`、`total_frames`、`meta_fps`、`meta_fps_source`、`device`、`clip_length`、`stride`、`resolution` |
| 失败形式 | 如果 side runtime、checkpoint 或 FPS 元数据不可用，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 这三个指标只要产出数值分数，当前都属于 `official` |
| 论文可用性结论 | `仅作辅助` |

- 这组指标的价值在于：它测到了普通 flow 和 smoothness 指标覆盖不到的一类失败模式，也就是“不合理的物理节奏”。
- 但它仍然只能算辅助性指标，因为分数依赖一个外部预训练节奏估计器，以及可信的 FPS 元数据。
- 如果你只是重新编码了预测视频，导致 container FPS 变了，即便帧内容没变，分数也可能变化。
- 建议用法：当时间标定很重要时报告它们，但不要把它们当成 motion-control 或整体质量的替代品。
- 放到更宽的 `real_time` 支柱里看，这组指标只是当前公开 benchmark 的一种实例化。未来的在线或交互式 track 还可以在同一支柱下继续加入 latency、throughput 和 wall-clock responsiveness。
