# 实时吞吐与动作延迟

Real-time evaluation 在 WorldArena 里不只是在问“每秒能生成多少帧”。对 interactive world model 来说，真正影响体验的是两个门槛：

!!! abstract "一句话读法"

    `effective_generation_fps` 和 `realtime_factor` 判断模型是否追得上播放；`action_response_latency` 和 `latency_budget_pass_rate` 判断用户动作是否能足够快地反映到画面中。

Temporal calibration 仍属于 Real Time 维度，但它回答的是“画面运动是否拥有合理的物理时钟”。本页的四个指标回答的是“系统是否能形成实时交互体验”。

## 指标地图

| 问题 | 指标 | 后端 | 越好方向 | 原始值 |
| --- | --- | --- | --- | --- |
| 生成是否追得上播放 | `effective_generation_fps` | `generation_metadata` | 越大越好 | 每秒真实生成的帧数 |
| 生成耗时是否短于视频时长 | `realtime_factor` | `generation_metadata` | 越大越好 | 输出视频时长 / 真实生成耗时 |
| 输入变化多久可见 | `action_response_latency` | `action_flow` | 越小越好 | action 到可见响应的平均延迟 |
| 响应是否稳定落在预算内 | `latency_budget_pass_rate` | `action_flow` | 越大越好 | 在延迟预算内响应的 transition 占比 |

!!! note "为什么要拆成 throughput 和 latency"

    Throughput 是必要条件，但不是充分条件。一个模型可以离线生成得很快，却在动作改变后过很多帧才响应；这种系统看起来有高 FPS，但实际玩起来不跟手。反过来，一个模型可能很快响应输入，但整体生成速度低于播放速度，也无法持续实时运行。

## 生成吞吐

吞吐指标只依赖 prediction video 和 generation timing metadata，不需要 action schedule。运行时会在 prediction 目录和若干上级目录中查找 `generation_records*.jsonl`，并用以下方式匹配 record：

| 匹配方式 | 字段 |
| --- | --- |
| 精确路径匹配 | `prediction_path` 或 `output_path` |
| 样本级回退匹配 | `sample_id`，且 suite 兼容 |

真实生成耗时按优先级读取：`generation_wall_time_seconds`、`wall_time_seconds`、`elapsed_seconds`、`runtime_seconds`。同时，WorldArena 会用 OpenCV probe prediction video，得到 `frame_count`、编码 `fps` 和 `duration_seconds`。

### `effective_generation_fps` {#effective-generation-fps}

该指标衡量系统实际产帧速度：

\[
\mathrm{effective\_generation\_fps}
=
\frac{\mathrm{frame\_count}}
{\mathrm{generation\_wall\_time\_seconds}}
\]

当 `target_fps` 可用时，归一化为：

\[
score =
\mathrm{clamp}\left(
\frac{\mathrm{effective\_generation\_fps}}
{\mathrm{target\_fps}},
0,
1
\right)
\]

`target_fps` 的解析优先级是：runtime config、generation record 的 `target_fps`、`output_fps`、`fps`、sample FPS，最后是视频 probe 得到的 FPS。

| 原始值 | 含义 |
| --- | --- |
| `>= target_fps` | 产帧速度达到播放目标 |
| `< target_fps` | 无法持续实时播放，需要等待、缓存或降帧 |
| `null` | 缺少 generation timing metadata，记为 `not_applicable` |

### `realtime_factor` {#realtime-factor}

该指标直接比较“生成出来的视频能播放多久”和“生成它花了多久”：

\[
\mathrm{realtime\_factor}
=
\frac{\mathrm{output\_duration\_seconds}}
{\mathrm{generation\_wall\_time\_seconds}}
\]

归一化使用 `[0, 1]` 正向缩放。

| 原始值 | 含义 |
| --- | --- |
| `1.0` | 生成刚好追上播放 |
| `< 1.0` | 慢于实时 |
| `> 1.0` | 快于播放；normalized score 仍封顶为 `1.0` |

!!! tip "两个吞吐指标怎么一起看"

    `effective_generation_fps` 更像系统吞吐；`realtime_factor` 更像最终播放体验。正常情况下二者趋势一致。如果出现分歧，通常要检查视频编码 FPS、frame count、target FPS 或 generation record 是否记录正确。

## 动作延迟

动作延迟指标需要 action payload。payload 可以直接写在 generation record 的 `action_payload` 字段中，也可以通过 `action_spec_path` 指向外部文件。

当前实现面向 Matrix-Game-style action schedule。它读取 `keyboard_condition` 和 `mouse_condition`，当 action vector 变化时切分 segment。第一个 segment 被视为初始状态，后续 segment 才作为候选 action transition。

### 可见响应检测流程

1. 解码 prediction video 为灰度帧，并按 `max_analysis_width` 下采样。
2. 根据 action transition 推导预期响应方向。
3. 对相邻帧计算 Farneback optical flow。
4. 用 transition 前若干帧估计 baseline flow。
5. 当预期方向上的 flow signal 超过 baseline-adjusted threshold 时，记录首个响应帧。
6. 若在 timeout 内没有检测到响应，该事件记为 missing response，并赋值为 `latency_timeout_seconds`。

| Action 来源 | 预期响应 proxy |
| --- | --- |
| vertical mouse movement | horizontal optical-flow sign |
| left/right keyboard channels | horizontal optical-flow sign |
| forward/back keyboard channels | radial optical-flow sign |

### `action_response_latency` {#action-response-latency}

对每个可测 action transition \(j\)：

\[
\ell_j =
\frac{\mathrm{response\_frame}_j - \mathrm{action\_start\_frame}_j}
{\mathrm{fps}}
\]

视频级 raw value 是所有可测 transition 的平均延迟：

\[
\mathrm{action\_response\_latency}
=
\frac{1}{N}\sum_{j=1}^{N}\ell_j
\]

归一化在 `[0, latency_threshold_seconds]` 上做反向缩放，默认阈值为 `0.1s`：

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

该指标统计有多少 transition 的响应延迟落在预算内：

\[
\mathrm{latency\_budget\_pass\_rate}
=
\frac{1}{N}
\sum_{j=1}^{N}
\mathbb{1}[\ell_j \le \tau]
\]

其中 \(\tau = \mathrm{latency\_threshold\_seconds}\)。归一化在 `[0, 1]` 上做恒等映射。

!!! tip "为什么 pass rate 也重要"

    平均延迟可能掩盖偶发卡顿。`action_response_latency` 低但 `latency_budget_pass_rate` 不高，说明模型多数时候响应快，但有些输入会明显掉链子；这种不稳定在交互里非常容易被用户感知。

## 配置项

默认运行参数位于 `benchmark.metric_runtime.real_time`：

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `latency_threshold_seconds` | `0.1` | action-latency pass/fail 的响应预算 |
| `latency_timeout_seconds` | `1.0` | 未检测到响应时赋予的延迟 |
| `max_analysis_width` | `320` | optical-flow latency 分析时的宽度上限 |
| `min_response_signal` | `0.04` | 判定响应所需的最小方向性 flow signal |
| `baseline_window_frames` | `4` | transition 前用于估计 baseline flow 的帧间隔数量 |

`target_fps` 从 benchmark config 注入，用于 `effective_generation_fps` 的归一化。

## 输入输出契约

| 项目 | 吞吐指标 | 动作延迟指标 |
| --- | --- | --- |
| 预测输入 | prediction video path | prediction video path |
| 必需 metadata | generation wall time | action payload 和 FPS |
| 参考输入 | 无 | 无 |
| 标注输入 | 无 | frame-level action schedule |
| 聚合状态 | timing metadata 可用时 official | 存在可测 action transition 时 official |
| not applicable 条件 | 缺少 generation timing metadata | 缺少 action spec，或没有可测 transition |
| details 字段 | record 是否匹配、wall time、frame count、output FPS、target FPS、duration | action count、measured events、latency mean/p50/p95/max、missing response count、per-event details |

## 解读矩阵

| 现象 | 可能含义 |
| --- | --- |
| 高吞吐 + 低延迟 | 最接近真正实时交互体验 |
| 高吞吐 + 高延迟 | 离线生成快，但输入响应不跟手 |
| 低吞吐 + 低延迟 | 响应方向可能正确，但无法持续实时播放 |
| `action_response_latency` 低 + `latency_budget_pass_rate` 低 | 平均值好看，但响应稳定性不足 |
| 吞吐指标为 `not_applicable` | 缺 generation timing metadata，不代表模型慢 |
| 延迟指标为 `not_applicable` | 缺 action schedule，常见于纯 image/text-conditioned generation |

## 适用边界

- 这组 latency metrics 是 visual proxy，衡量可见运动响应，不是内部调度延迟或 GPU kernel latency。
- 它最适合带明确 action conditioning、且动作会引起清晰相机或自运动响应的输出，例如 Matrix-Game rollout。
- 对 YuMe 这类 image/text-conditioned generation，吞吐指标仍然可用；如果没有逐帧 action schedule，动作延迟指标应保持 `not_applicable`。
