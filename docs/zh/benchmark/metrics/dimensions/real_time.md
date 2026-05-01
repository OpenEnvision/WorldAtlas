# Real Time

`real_time` 维度衡量视频世界模型是否能支撑真人交互体验。WorldArena 现在把它拆成两个核心问题：

- **Throughput**：模型生成帧的速度，是否至少赶得上用户消费帧的速度。
- **Latency**：交互动作变化后，生成画面需要多久才出现可见响应。

Temporal calibration 仍然属于这个维度，但它回答的是另一个问题：画面运动本身是否拥有物理上合理的时间节奏。

## 范围

| 层级 | 指标 |
| --- | --- |
| 生成吞吐 | `effective_generation_fps`、`realtime_factor` |
| 动作延迟 | `action_response_latency`、`latency_budget_pass_rate` |
| 平均物理 FPS 误差 | [`physical_fps_error`](../temporal_calibration.md) |
| 相对 FPS 误差 | [`physical_fps_pct_error`](../temporal_calibration.md) |
| 视频内部节奏稳定性 | [`physical_fps_intra_video_cv`](../temporal_calibration.md) |

## 指标定义

`effective_generation_fps` 衡量系统实际吞吐：

\[
\mathrm{effective\_generation\_fps}
=
\frac{\mathrm{generated\_frame\_count}}{\mathrm{generation\_wall\_time\_seconds}}
\]

`realtime_factor` 比较输出视频时长和真实生成耗时：

\[
\mathrm{realtime\_factor}
=
\frac{\mathrm{output\_duration\_seconds}}{\mathrm{generation\_wall\_time\_seconds}}
\]

`1.0` 表示生成速度刚好追上播放速度；低于 `1.0` 则无法持续实时消费。

`action_response_latency` 衡量 action transition 到首次可见响应帧之间的平均延迟。当前实现用 optical-flow 方向作为轻量 proxy，判断画面运动是否开始匹配动作。`latency_budget_pass_rate` 则统计有多少 action transition 能在配置的延迟预算内响应；默认预算是面向游戏式交互的 `0.1s`。

## 读分注意

- Throughput 是必要条件，但不是充分条件：模型每秒生成帧数足够高，仍然可能因为 action latency 太高而显得不跟手。
- Action latency 只适用于 generation record 中带有 action spec 的模型，例如 Matrix-Game 的 action payload。
- YuMe 这类 image/text-conditioned 生成可以报告吞吐指标；如果没有逐帧交互动作表，则 action-latency 指标会标记为 not applicable。
- 现有 `physical_fps_*` 指标仍然用于时间节奏标定，但它们不衡量用户输入延迟。
