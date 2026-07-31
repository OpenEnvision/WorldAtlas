# 指标总览

本文按 `config/benchmark.yaml` 中的公开 benchmark 面组织。四个 formal suite 是公开 benchmark 主体；`experimental` 是补充切片，不应混成 benchmark 总分。

## 读数规则

聚合报告中的字段含义如下：

- **`raw`** — 后端原始值；可能是标量或分量字典
- **`normalized`** — 写入聚合的 `[0, 1]` 分数；统一约定越高越好
- **`official`** — 进入 `per_metric_summary.json` 和 `per_dimension_summary.json`
- **`diagnostic`** — 非 official；永远不进 headline 聚合
- **`not_applicable`** — 强监督不足，不计算、不聚合；进入 `not_applicable_summary.json`
- **`normalized = null`** — runtime 失败或缺输入；不会被自动记为 0

各指标的归一化方式、适用 suite 与 official 门槛，见对应指标页。

## Suite 向量

| Suite | 指标数 | 说明 |
| --- | ---: | --- |
| `image_static` | 43 | 以 image 为条件的静态世界生成；video prediction 在输入允许时按 video 指标面扩展评测 |
| `image_dynamic` | 33 | 同 `image_static`，通常覆盖更多 motion prompt |
| `video_static` | 37 | 以 video 为条件的静态世界生成 |
| `video_dynamic` | 44 | 以 video 为条件的动态世界生成 |
| `experimental` | 3 | 补充切片，不参与 formal headline 汇总 |

哪些指标在某个 suite 中为 official、diagnostic 或 `not_applicable`，取决于监督信号与后端是否成功；细节见各指标页与 [基准评测总览](../index.md)。

## 六个评测支柱

Metrics 导航按六个顶层维度组织；每个维度有独立索引页，汇总该维度的指标族与解读重点。

| 维度 | 分类页 |
| --- | --- |
| Long sequence（memory） | [Long Sequence（Memory）](dimensions/long_sequence.md) |
| Action control | [Action Control](dimensions/action_control.md) |
| 3D/4D consistency | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| Physics | [Physics](dimensions/physics.md) |
| Quality | [Quality](dimensions/quality.md) |
| Real time | [Real Time](dimensions/real_time.md) |

## 指标索引

下列链接按维度分组。算法、后端、归一化与聚合状态见各指标页。

**Action control** — [`prompt_alignment`](prompt_alignment.md) · [`semantic_alignment`](semantic_alignment.md) · [`camera_error`](camera_error.md) · [`motion_accuracy`](motion_accuracy.md)

**Quality** — [`style_consistency`](style_consistency.md) · [`perceptual_quality`](perceptual_quality.md) · [`image_quality`](image_quality.md) · [`hpsv3_norm`](hpsv3_norm.md) · [`brightness_distribution_consistency`](brightness_consistency.md) · [`color_temperature_constraint`](color_temperature_consistency.md) · [`sharpness_retention`](sharpness_retention.md) · [`psnr`](psnr.md) · [`ssim`](ssim.md) · [`long_sequence_aesthetic_quality`](long_sequence.md) · [`long_sequence_imaging_quality`](long_sequence.md)

**Long sequence（memory）** — [`seen_region_preservation`](seen_region_preservation.md) · [`object_permanence`](object_permanence.md) · [`memory_revisit_consistency`](memory_revisit_consistency.md) · [`long_sequence_motion_smoothness`](long_sequence.md)

**3D/4D consistency** — [`novel_region_fidelity`](novel_region_fidelity.md) · [`reprojection_error`](reprojection_error.md) · [`optical_flow_aepe`](optical_flow_aepe.md) · [`epipolar_error`](gvgc_geometry.md#epipolar-error) · [`epipolar_inlier_rate`](gvgc_geometry.md#epipolar-inlier-rate) · [`feature_match_coverage`](gvgc_geometry.md#feature-match-coverage) · [`depth_accuracy`](depth_accuracy.md) · [`depth_collision`](depth_collision.md) · [`met3r_consistency`](met3r_consistency.md) · [`reconstruction_consistency`](reconstruction_consistency.md)

**Physics** — [`pmf`](pmf.md) · [`motion_magnitude`](motion_magnitude.md) · [`motion_smoothness`](motion_smoothness.md) · [`long_sequence_dynamic_degree`](long_sequence.md)

**Real time** — [`physical_fps_error`](temporal_calibration.md) · [`physical_fps_pct_error`](temporal_calibration.md) · [`physical_fps_intra_video_cv`](temporal_calibration.md) · [`effective_generation_fps`](realtime_throughput_latency.md#effective-generation-fps) · [`realtime_factor`](realtime_throughput_latency.md#realtime-factor) · [`action_response_latency`](realtime_throughput_latency.md#action-response-latency) · [`latency_budget_pass_rate`](realtime_throughput_latency.md#latency-budget-pass-rate)

## 独立 Physics Track

`config/benchmark_physics.yaml` 是 simulator-backed physics 配置，和默认 `config/benchmark.yaml` 的 `experimental` 不是同一个评测面。它携带 simulator-backed 物理指标，并统一汇总到顶层 `physics` 维度。指标语义、artifact 需求与可比性以 [Physics](dimensions/physics.md)、[Simulator-Backed Physics](physics/index.md) 与 [Physics Metrics & Contracts](physics/metrics.md) 为准。
