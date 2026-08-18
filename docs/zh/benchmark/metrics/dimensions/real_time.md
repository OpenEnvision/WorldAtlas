# Real Time

`real_time` 维度衡量视频世界模型是否能支撑真人交互体验。WorldArena 将其拆分为吞吐、动作延迟与时间标定三个互补问题，并将维度汇总结果写入 `per_dimension_summary.json`。核心问题是生成能否跟上播放，并及时响应用户动作。

### 范围

本维度围绕实时世界生成的三个互补问题展开。生成吞吐关注模型生成帧的速度是否至少赶得上用户消费帧的速度，主要由 [`effective_generation_fps`](../realtime_throughput_latency.md#effective-generation-fps) 和 [`realtime_factor`](../realtime_throughput_latency.md#realtime-factor) 衡量。

动作延迟关注交互动作发生变化后，生成画面需要多久才出现可见响应，主要由 [`action_response_latency`](../realtime_throughput_latency.md#action-response-latency) 和 [`latency_budget_pass_rate`](../realtime_throughput_latency.md#latency-budget-pass-rate) 衡量。Temporal calibration 则关注画面运动本身是否拥有物理上合理的时间节奏，包括 [`physical_fps_error`](../temporal_calibration.md)、[`physical_fps_pct_error`](../temporal_calibration.md) 和 [`physical_fps_intra_video_cv`](../temporal_calibration.md)。

### 指标定义

[`effective_generation_fps`](../realtime_throughput_latency.md#effective-generation-fps) 和 [`realtime_factor`](../realtime_throughput_latency.md#realtime-factor) 的详细定义见[实时吞吐与动作延迟](../realtime_throughput_latency.md)，用于衡量生成过程是否能追上播放速度。[`action_response_latency`](../realtime_throughput_latency.md#action-response-latency) 和 [`latency_budget_pass_rate`](../realtime_throughput_latency.md#latency-budget-pass-rate) 的详细定义也见该页面，它们在存在逐帧 action schedule 时衡量 action 到可见响应的延迟。

### 解读边界

Throughput 是必要条件，但不是充分条件：模型每秒生成帧数足够高，仍然可能因为 action latency 太高而显得不跟手。Action latency 只适用于 generation record 中带有 action spec 的模型，例如 Matrix-Game 的 action payload。YuMe 这类 image/text-conditioned 生成可以报告吞吐指标；如果没有逐帧交互动作表，则 action-latency 指标会标记为 `not_applicable`。

现有 `physical_fps_*` 指标仍然用于时间节奏标定，但它们不衡量用户输入延迟。
