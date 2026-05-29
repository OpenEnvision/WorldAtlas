# Realtime Throughput and Latency

Real-time evaluation in WorldArena is not just a question of frames per second. For an interactive world model, the user experience depends on two gates:

!!! abstract "One-line reading"

    `effective_generation_fps` and `realtime_factor` ask whether generation keeps up with playback; `action_response_latency` and `latency_budget_pass_rate` ask whether user actions become visible quickly enough.

Temporal calibration still belongs to the Real Time dimension, but it answers a different question: whether the visual motion has a physically plausible clock. The four metrics on this page answer whether the system can support an interactive real-time loop.

## Metric Map

| Question | Metric | Backend | Direction | Raw value |
| --- | --- | --- | --- | --- |
| Can generation keep up with playback? | `effective_generation_fps` | `generation_metadata` | higher is better | generated frames per wall-clock second |
| Is generation faster than the video it produces? | `realtime_factor` | `generation_metadata` | higher is better | output duration divided by generation wall time |
| How long until an input becomes visible? | `action_response_latency` | `action_flow` | lower is better | mean action-to-visible-response delay |
| Are responses consistently inside budget? | `latency_budget_pass_rate` | `action_flow` | higher is better | fraction of transitions within the latency budget |

!!! note "Why split throughput and latency"

    Throughput is necessary but not sufficient. A model can generate offline frames quickly and still feel unresponsive if action changes take many frames to appear. Conversely, a model can react promptly to inputs but still fail real-time playback if generation is slower than consumption.

## Throughput

Throughput metrics only require the prediction video and generation timing metadata. The runtime searches for `generation_records*.jsonl` files in the prediction directory and nearby parent directories, then matches a record in one of two ways:

| Match mode | Fields |
| --- | --- |
| exact path match | `prediction_path` or `output_path` |
| sample fallback | `sample_id` with a compatible suite |

Wall-clock generation time is read, in priority order, from `generation_wall_time_seconds`, `wall_time_seconds`, `elapsed_seconds`, or `runtime_seconds`. WorldArena also probes the prediction video with OpenCV to recover `frame_count`, encoded `fps`, and `duration_seconds`.

### `effective_generation_fps` {#effective-generation-fps}

This metric measures the actual frame production rate:

\[
\mathrm{effective\_generation\_fps}
=
\frac{\mathrm{frame\_count}}
{\mathrm{generation\_wall\_time\_seconds}}
\]

When `target_fps` is available, normalization is:

\[
score =
\mathrm{clamp}\left(
\frac{\mathrm{effective\_generation\_fps}}
{\mathrm{target\_fps}},
0,
1
\right)
\]

`target_fps` is resolved from runtime config, generation record `target_fps`, generation record `output_fps`, generation record `fps`, sample FPS, and finally probed video FPS.

| Raw value | Meaning |
| --- | --- |
| `>= target_fps` | generation reaches the playback target |
| `< target_fps` | sustained real-time playback needs waiting, buffering, or dropping frames |
| `null` | generation timing metadata is missing, so the metric is `not_applicable` |

### `realtime_factor` {#realtime-factor}

This metric directly compares playable output duration with generation wall time:

\[
\mathrm{realtime\_factor}
=
\frac{\mathrm{output\_duration\_seconds}}
{\mathrm{generation\_wall\_time\_seconds}}
\]

Normalization uses forward scaling over `[0, 1]`.

| Raw value | Meaning |
| --- | --- |
| `1.0` | generation exactly keeps up with playback |
| `< 1.0` | slower than real time |
| `> 1.0` | faster than playback; normalized score is still capped at `1.0` |

!!! tip "How to read the two throughput metrics together"

    `effective_generation_fps` is closer to system throughput; `realtime_factor` is closer to final playback experience. They usually move together. If they disagree, inspect encoded FPS, frame count, target FPS, and generation record fields.

## Action Latency

Action-latency metrics require an action payload. The payload can be embedded in the generation record as `action_payload`, or referenced through `action_spec_path`.

The current implementation targets Matrix-Game-style action schedules. It reads `keyboard_condition` and `mouse_condition`, then splits them into segments whenever the action vector changes. The first segment is treated as the initial state; later segments are candidate action transitions.

### Visible-Response Detection

1. Decode the prediction video to grayscale frames and downsample to `max_analysis_width`.
2. Infer the expected response direction for each action transition.
3. Compute Farneback optical flow between consecutive frames.
4. Estimate baseline flow from the frames before the transition.
5. Record the first response frame whose flow signal exceeds the baseline-adjusted threshold.
6. If no response is detected before timeout, count the event as missing and assign `latency_timeout_seconds`.

| Action source | Expected response proxy |
| --- | --- |
| vertical mouse movement | horizontal optical-flow sign |
| left/right keyboard channels | horizontal optical-flow sign |
| forward/back keyboard channels | radial optical-flow sign |

### `action_response_latency` {#action-response-latency}

For each measurable action transition \(j\):

\[
\ell_j =
\frac{\mathrm{response\_frame}_j - \mathrm{action\_start\_frame}_j}
{\mathrm{fps}}
\]

The video-level raw value is mean latency:

\[
\mathrm{action\_response\_latency}
=
\frac{1}{N}\sum_{j=1}^{N}\ell_j
\]

Normalization is lower-is-better over `[0, latency_threshold_seconds]`, with `0.1s` as the default threshold:

\[
score =
1 -
\mathrm{clamp}\left(
\frac{raw}{\mathrm{latency\_threshold\_seconds}},
0,
1
\right)
\]

### `latency_budget_pass_rate` {#latency-budget-pass-rate}

This metric reports how often measured transitions respond inside the budget:

\[
\mathrm{latency\_budget\_pass\_rate}
=
\frac{1}{N}
\sum_{j=1}^{N}
\mathbb{1}[\ell_j \le \tau]
\]

where \(\tau = \mathrm{latency\_threshold\_seconds}\). Normalization is identity over `[0, 1]`.

!!! tip "Why pass rate matters"

    Mean latency can hide occasional stalls. Low `action_response_latency` with a weak `latency_budget_pass_rate` means most inputs respond quickly, but some input changes are visibly late; that instability is easy for users to feel.

## Configuration

Default runtime settings live under `benchmark.metric_runtime.real_time`:

| Key | Default | Meaning |
| --- | ---: | --- |
| `latency_threshold_seconds` | `0.1` | response budget for action-latency pass/fail |
| `latency_timeout_seconds` | `1.0` | assigned latency when no response is detected |
| `max_analysis_width` | `320` | downsample width cap for optical-flow latency analysis |
| `min_response_signal` | `0.04` | minimum directional flow signal for response detection |
| `baseline_window_frames` | `4` | number of pre-transition frame gaps used for baseline flow |

`target_fps` is injected from the benchmark config and used by `effective_generation_fps`.

## Input And Output Contract

| Item | Throughput metrics | Action-latency metrics |
| --- | --- | --- |
| Prediction input | prediction video path | prediction video path |
| Required metadata | generation wall time | action payload and FPS |
| Reference input | none | none |
| Annotation input | none | frame-level action schedule |
| Aggregation | official when timing metadata is available | official when measurable action transitions exist |
| Not applicable when | generation timing metadata is absent | action spec is absent or no transition can be measured |
| Details payload | record match flag, wall time, frame count, output FPS, target FPS, duration | action count, measured events, latency mean/p50/p95/max, missing response count, per-event details |

## Interpretation Matrix

| Observation | Likely meaning |
| --- | --- |
| high throughput + low latency | closest to a true interactive real-time experience |
| high throughput + high latency | fast offline generation, but sluggish input response |
| low throughput + low latency | the response direction may be right, but playback cannot be sustained |
| low `action_response_latency` + low `latency_budget_pass_rate` | good average, unstable response tail |
| throughput metrics are `not_applicable` | generation timing metadata is missing; this is not evidence that the model is slow |
| latency metrics are `not_applicable` | action schedule is missing, common for image/text-conditioned generation |

## Applicability

- These latency metrics are visual proxies. They measure visible motion response, not internal scheduling delay or GPU kernel latency.
- They work best for action-conditioned outputs with clear camera or ego-motion response, such as Matrix-Game rollouts.
- YuMe-style image/text-conditioned generation can still report throughput metrics; without a frame-level action schedule, action latency should remain `not_applicable`.
