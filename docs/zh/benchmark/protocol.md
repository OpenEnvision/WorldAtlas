# 协议与资格

## 采样契约

| 项目 | 值 | 为什么重要 |
| --- | --- | --- |
| 官方协议 | `segment_random` | headline run 会在对齐的时间支撑上采样参考与预测 |
| 兼容协议 | `anchor_compat` | 面向旧实验的遗留 anchor-only 路径 |
| 目标 FPS | `24` | 统一评测时的时间假设 |
| 锚点位置 | `0.0`、`0.3333`、`0.6667`、`1.0` | 锚点类指标会从 rollout window 的这些位置读取帧 |
| 运动帧数 | `12` | motion 类指标每个样本默认解码 12 帧窗口 |
| 分段数 | `4` | 在 `segment_random` 下控制确定性的时间段选择 |
| 随机种子 | `17` | 保持样本级采样稳定可复现 |

锚点类指标和 motion 类指标共享同一份 sample-local protocol state，因此 prediction 和 reference 会在对齐的时间支撑上比较。

## 条件输入契约

| 资产类型 | 条件输入策略 | 默认参数 | 对评测的影响 |
| --- | --- | --- | --- |
| image suite | `reference_image` | `conditioning_frame_count = 1` | 一张静态参考图像作为条件输入；prediction 按 video rollout 解析，因此可运行预测侧视频指标 |
| `video_static` | `first_frame` | `conditioning_frame_count = 1` | 首帧给定，后续 rollout 被视为 camera-conditioned continuation |
| `video_dynamic` | `reference_video` | `dynamic_prefix_ratio = 0.25`、`dynamic_prefix_min_frames = 4` | rollout 前缀作为条件输入，后续 continuation 区域作为主要评测目标 |

manifest 会把这个契约写到每个样本里，因此下游生成和评测不需要再靠目录结构反推。

## Eligibility 状态

| 状态 | 是否进入 official 聚合 | 典型原因 |
| --- | --- | --- |
| `official` | 是 | 指标产出了数值，且样本满足监督门槛 |
| `diagnostic` | 否 | 指标不进 headline 聚合 |
| `not_applicable` | 否 | 样本没有足够强的监督来支撑该指标结论，因此不计算、不聚合 |
| `normalized = null` | 否 | runtime 失败、输入缺失或指标不适用 |

## 指标门槛

| 指标或指标族 | official 门槛 |
| --- | --- |
| `prompt_alignment` | 样本必须有显式 instruction supervision，并且不能是 self-conditioned |
| `camera_error` | 需要 `sample.annotation_path` 下存在姿态标注 |
| `motion_accuracy` | 需要对齐好的 segmentation mask |
| `object_permanence` | 需要对齐好的动态 mask，以及可读取的参考视频 |
| `physical_fps_*` | 需要可用的 `Pulse-of-Motion` 运行时，以及可读取的 container FPS 或 manifest FPS |
| `met3r_consistency` | reference/prediction 锚点可读取时进入 official；空 overlap mask 会被记录并计为最低归一化分（`0`），而不是从均值中排除 |
| `depth_collision` | 永远不进 headline 聚合。需要 video prediction 和 Video-Depth-Anything runtime |
| `reconstruction_consistency` | 永远不进 headline 聚合。需要 video prediction 与 Depth-Anything-3 深度/位姿重投影运行时 |
| `memory_revisit_consistency` | 永远不进 headline 聚合。只适合回访/闭环类 experimental 样本 |
| `semantic_alignment`、`depth_accuracy`、`long_sequence_*` | 永远不进 headline 聚合 |
| formal suite 中其余指标 | 只要成功返回数值，就属于 official |

两个常见后果：

- 自条件或缺少显式 instruction 的样本不会计算 `prompt_alignment`，而是进入 `not_applicable_summary.json`。
- `image_static` / `image_dynamic` 的输出是 I2V 视频，所以 prediction-only 视频指标会被正常计算；需要 reference video、pose 或 mask 的指标会在 sidecar 不足时返回空值，而不是强行打分。

## 归一化

所有 headline 值都以 `[0, 1]` 范围内的 `normalized` 分数保存。

| 指标家族 | 原始形式 | 归一化规则 |
| --- | --- | --- |
| 有界标量指标 | 本身有界、或自然可映射到有界区间的单标量 | 来自 `config/benchmark.yaml` 的直接 min/max 缩放 |
| 经验标量指标 | “越小越好”或“越大越好”的单标量 | 经验 min/max 缩放，可选反向 |
| z-score 指标 | 相对 benchmark 统计量校准过的单标量 | 把 z-score 映射到 `[0, 1]` 的配置子区间 |
| 组件型指标 | 多个原始分量 | 各分量分别归一化后再做 reduce |

当前 benchmark 特有的 reducer：

- `camera_error`：分别归一化 rotation 和 translation，再用几何平均做归约。
- `motion_smoothness`：分别归一化 `MSE`、`SSIM` 和 `LPIPS`，再用算术平均做归约。

## 聚合规则

- `per_metric_summary.json` 只会平均那些 `eligibility_status = official` 的数值型 `normalized` 分数。
- `per_dimension_summary.json` 会把同一批 official 分数重新汇总到六个顶层维度：`long_sequence`、`action_control`、`consistency_3d_4d`、`physics`、`quality`、`real_time`。
- `diagnostic_metric_summary.json` 只平均明确标记为非 official 的数值型分数。
- `not_applicable_summary.json` 记录因强监督不足而没有计算的指标；这些条目不是失败，也不是非 official 分数。
- `domain_breakdown.json` 也只统计 official 分数，并按 `style`、`environment`、`scene` 和 `motion_category` 切分。
- 缺失预测会在 `run_manifest.json` 中被计数，但不会被自动当作 0 分。它们就是缺失。
- `experimental` 不是 formal headline suite；当前实现中 `motion_magnitude` 和 `motion_smoothness` 如果在该 suite 成功产出，会按默认 `official` 路由写入报告。

simulator-backed physics 配置会复用同一套 report contract，但它会落到独立 manifest 和单独的配置 namespace 上。需要注意的是，physics 配置同样使用了 suite 名 `experimental`，但这里指的是 simulator-backed physics track，而不是 `config/benchmark.yaml` 里那个仅作补充的 `experimental` 切片。详见 Metrics 下的 [Physics](metrics/dimensions/physics.md)。

只有在 suite、protocol、metric config 和预测完整度都一致时，两个 run 才可直接比较。WorldAtlas Arena 当前只有一个测试集；manifest 中的 `split` 字段固定为 `test`，仅作为兼容字段保留。
