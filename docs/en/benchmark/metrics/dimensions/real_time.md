# Real Time

The `real_time` dimension covers whether a video world model can support an interactive human experience. WorldArena now separates this into two questions:

- **Throughput:** can the model generate frames at least as fast as users consume them?
- **Latency:** after an interactive action changes, how long does it take for the generated frames to visibly respond?

Temporal calibration remains part of this dimension, but it answers a different question: whether the visual motion has a physically plausible clock.

## Scope

| Layer | Metrics |
| --- | --- |
| generation throughput | `effective_generation_fps`, `realtime_factor` |
| action latency | `action_response_latency`, `latency_budget_pass_rate` |
| average physical FPS error | [`physical_fps_error`](../temporal_calibration.md) |
| relative FPS error | [`physical_fps_pct_error`](../temporal_calibration.md) |
| intra-video rhythm stability | [`physical_fps_intra_video_cv`](../temporal_calibration.md) |

## Metric Definitions

`effective_generation_fps` measures raw system throughput:

\[
\mathrm{effective\_generation\_fps}
=
\frac{\mathrm{generated\_frame\_count}}{\mathrm{generation\_wall\_time\_seconds}}
\]

`realtime_factor` compares generated video duration with wall-clock generation time:

\[
\mathrm{realtime\_factor}
=
\frac{\mathrm{output\_duration\_seconds}}{\mathrm{generation\_wall\_time\_seconds}}
\]

A value of `1.0` means the model generates as fast as playback. Values below `1.0` cannot sustain real-time consumption.

`action_response_latency` measures the average delay between an action transition and the first generated frame whose optical-flow direction matches that action. `latency_budget_pass_rate` reports the fraction of measured action transitions that respond within the configured latency budget, currently `0.1s` by default for game-like interaction.

## Reading Notes

- Throughput is necessary but not sufficient: a model can generate enough frames per second while still feeling unresponsive if action latency is high.
- Action latency is only applicable when generation records include an action spec, such as Matrix-Game action payloads.
- YuMe-style image/text-conditioned generation can report throughput metrics, but action-latency metrics are not applicable unless a frame-level interaction schedule is available.
- The existing `physical_fps_*` metrics remain useful for timing calibration, but they do not measure user input latency.
