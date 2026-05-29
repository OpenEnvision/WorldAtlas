# Real Time

The `real_time` dimension covers whether a video world model can support an interactive human experience. WorldArena now separates this into two questions:

- **Throughput:** can the model generate frames at least as fast as users consume them?
- **Latency:** after an interactive action changes, how long does it take for the generated frames to visibly respond?

Temporal calibration remains part of this dimension, but it answers a different question: whether the visual motion has a physically plausible clock.

## Scope

| Layer | Metrics |
| --- | --- |
| generation throughput | [`effective_generation_fps`](../realtime_throughput_latency.md#effective-generation-fps), [`realtime_factor`](../realtime_throughput_latency.md#realtime-factor) |
| action latency | [`action_response_latency`](../realtime_throughput_latency.md#action-response-latency), [`latency_budget_pass_rate`](../realtime_throughput_latency.md#latency-budget-pass-rate) |
| average physical FPS error | [`physical_fps_error`](../temporal_calibration.md) |
| relative FPS error | [`physical_fps_pct_error`](../temporal_calibration.md) |
| intra-video rhythm stability | [`physical_fps_intra_video_cv`](../temporal_calibration.md) |

## Metric Definitions

`effective_generation_fps` and `realtime_factor` are detailed in [Realtime Throughput and Latency](../realtime_throughput_latency.md). They measure whether generation keeps up with playback.

`action_response_latency` and `latency_budget_pass_rate` are also detailed in [Realtime Throughput and Latency](../realtime_throughput_latency.md). They measure action-to-visible-response delay when frame-level action schedules are available.

## Reading Notes

- Throughput is necessary but not sufficient: a model can generate enough frames per second while still feeling unresponsive if action latency is high.
- Action latency is only applicable when generation records include an action spec, such as Matrix-Game action payloads.
- YuMe-style image/text-conditioned generation can report throughput metrics, but action-latency metrics are not applicable unless a frame-level interaction schedule is available.
- The existing `physical_fps_*` metrics remain useful for timing calibration, but they do not measure user input latency.
