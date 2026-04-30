# 指标总览

本文按 `config/benchmark.yaml` 中的公开 benchmark 面组织。四个 formal suite 是公开 benchmark 主体；`experimental` 是补充切片，不应混成 benchmark 总分。

## 读数规则

| 字段 | 含义 |
| --- | --- |
| `raw` | 后端原始值；可能是标量或分量字典 |
| `normalized` | 写入聚合的 `[0, 1]` 分数；统一约定越高越好 |
| `official` | 进入 `per_metric_summary.json` 和 `per_dimension_summary.json` |
| `diagnostic` | 只作诊断；永远不进 headline 聚合 |
| `not_applicable` | 强监督不足，不计算、不聚合；进入 `not_applicable_summary.json` |
| `normalized = null` | runtime 失败或缺输入；不会被自动记为 0 |

## Suite 向量

| Suite | 指标数 | official / 条件 official | diagnostic / not applicable |
| --- | ---: | --- | --- |
| `image_static` | 31 | image-to-video 输出按 video prediction 扩展评测：`style_consistency`、`perceptual_quality`、`reprojection_error`、`optical_flow_aepe`、`motion_magnitude`、`motion_smoothness`、`physical_fps_*` 成功即 official；`prompt_alignment`、`object_detection`、`camera_error`、`motion_accuracy` 受监督门槛约束；`seen_region_preservation`、`novel_region_fidelity`、`object_permanence`、`pmf` 需要兼容 reference video / mask，否则通常为空 | `semantic_alignment`、`brightness_consistency`、`color_temperature_consistency`、`sharpness_retention`、`depth_accuracy`、`depth_collision`、`psnr`、`ssim`、`met3r_consistency`、`reconstruction_consistency`、`long_sequence_motion_smoothness`、`long_sequence_dynamic_degree`、`long_sequence_aesthetic_quality`、`long_sequence_imaging_quality` |
| `image_dynamic` | 31 | 同 `image_static`，但 motion prompt / object prompt 覆盖通常更高 | 同 `image_static` |
| `video_static` | 19 | `style_consistency`、`perceptual_quality`、`seen_region_preservation`、`novel_region_fidelity`、`reprojection_error`、`optical_flow_aepe`；`prompt_alignment`、`object_detection`、`camera_error` 受监督门槛约束 | `semantic_alignment`、`brightness_consistency`、`color_temperature_consistency`、`sharpness_retention`、`depth_accuracy`、`depth_collision`、`psnr`、`ssim`、`met3r_consistency`、`reconstruction_consistency` |
| `video_dynamic` | 25 | `style_consistency`、`perceptual_quality`、`object_permanence`、`pmf`、`motion_magnitude`、`motion_smoothness`、`physical_fps_*`；`prompt_alignment`、`motion_accuracy` 受监督门槛约束 | `semantic_alignment`、`brightness_consistency`、`color_temperature_consistency`、`sharpness_retention`、`depth_accuracy`、`depth_collision`、`psnr`、`ssim`、`met3r_consistency`、`reconstruction_consistency`、`long_sequence_motion_smoothness`、`long_sequence_dynamic_degree`、`long_sequence_aesthetic_quality`、`long_sequence_imaging_quality` |
| `experimental` | 3 | 当前实现中 `motion_magnitude`、`motion_smoothness` 成功即写入 official；该 suite 非 formal headline | `memory_revisit_consistency` |

## 六个评测支柱

Metrics 导航现在按这六个顶层维度组织；每个维度都有自己的索引页。

| Key | Label | 覆盖问题 | 分类页 |
| --- | --- | --- | --- |
| `long_sequence` | Long sequence（memory） | 状态记忆、外观保持与长时序 rollout 稳定性 | [Long Sequence（Memory）](dimensions/long_sequence.md) |
| `action_control` | Action control | scene control、camera control、object/action control | [Action Control](dimensions/action_control.md) |
| `consistency_3d_4d` | 3D/4D consistency | 几何、多视角、重投影与运动场一致性 | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| `physics` | Physics | 运动物理真实性与 simulator-backed physics 指标 | [Physics](dimensions/physics.md) |
| `quality` | Quality | 风格、感知质量、低层外观稳定性与重建式保真 | [Quality](dimensions/quality.md) |
| `real_time` | Real time | 节奏与 FPS 标定 | [Real Time](dimensions/real_time.md) |

## 完整指标清单

| 指标 | 维度 | 后端 | Suite | 原始值与归一化 | 聚合状态 |
| --- | --- | --- | --- | --- | --- |
| [`prompt_alignment`](prompt_alignment.md) | `action_control` | `reference` | formal 4 suite | 对 video prediction 的锚点帧计算 CLIPScore 类 `[0,1]` 标量；恒等缩放 | 显式 instruction 且非 self-conditioned 时 official，否则 `not_applicable`；非视频 prediction 是提交错误 |
| [`semantic_alignment`](semantic_alignment.md) | `action_control` | `clip` | formal 4 suite | 预测帧 vs 参考 caption/instruction 的 CLIP 相似度 | diagnostic |
| [`object_detection`](object_detection.md) | `action_control` | `grounding_dino` | image 2 suite、`video_static` | object prompt 检测成功率；`[0,1]` | 显式 object 标注 official；caption 推导不打分，记为 `not_applicable` |
| [`camera_error`](camera_error.md) | `action_control` | `droid_slam` | `image_static`、`image_dynamic`、`video_static` | `{rotation_error, translation_error}`；几何平均，越小越好 | 有 pose 标注且成功时 official |
| [`motion_accuracy`](motion_accuracy.md) | `action_control` | `sam2_sea_raft` | `image_static`、`image_dynamic`、`video_dynamic` | mask 内 motion accuracy z-score 映射 | 有动态 mask 且成功时 official |
| [`style_consistency`](style_consistency.md) | `quality` | `reference` | formal 4 suite | VGG Gram style similarity；`[0,1]` | official |
| [`perceptual_quality`](perceptual_quality.md) | `quality` | `reference` | formal 4 suite | 参考质量后端标量；`[0,1]` | official |
| [`brightness_consistency`](brightness_consistency.md) | `quality` | `luminance_temporal` | formal 4 suite | 亮度时序稳定性代理分数；`[0,1]` | diagnostic |
| [`color_temperature_consistency`](color_temperature_consistency.md) | `quality` | `hsv_hue_drift` | formal 4 suite | HSV hue drift 稳定性代理分数；`[0,1]` | diagnostic |
| [`sharpness_retention`](sharpness_retention.md) | `quality` | `tenengrad_brisque` | formal 4 suite | BRISQUE gate 约束下的 Tenengrad 清晰度保持；`[0,1]` | diagnostic |
| [`psnr`](psnr.md) | `quality` | `framewise` | formal 4 suite | paired-frame PSNR；`[10,40]` 正向缩放 | diagnostic |
| [`ssim`](ssim.md) | `quality` | `framewise` | formal 4 suite | paired-frame SSIM；`[0,1]` | diagnostic |
| [`long_sequence_aesthetic_quality`](long_sequence.md) | `quality` | `long_sequence` | `image_static`、`image_dynamic`、`video_dynamic` | Long-Sequence aesthetic quality；`[0,1]` | diagnostic |
| [`long_sequence_imaging_quality`](long_sequence.md) | `quality` | `long_sequence` | `image_static`、`image_dynamic`、`video_dynamic` | Long-Sequence imaging quality；`[0,1]` | diagnostic |
| [`seen_region_preservation`](seen_region_preservation.md) | `long_sequence` | `flow_dino` | `image_static`、`image_dynamic`、`video_static` | first-frame seen mask 内 PSNR/DINO 组合分 | 成功时 official；纯 image reference 通常为空 |
| [`object_permanence`](object_permanence.md) | `long_sequence` | `mask_dino` | `image_static`、`image_dynamic`、`video_dynamic` | dynamic mask 内 reference/pred identity 曲线匹配 | 有 video reference + mask 且成功时 official |
| [`memory_revisit_consistency`](memory_revisit_consistency.md) | `long_sequence` | `dinov2` | `experimental` | 闭环相似度 × 中途进展门控；`[0,1]` | diagnostic |
| [`long_sequence_motion_smoothness`](long_sequence.md) | `long_sequence` | `long_sequence` | `image_static`、`image_dynamic`、`video_dynamic` | Long-Sequence motion smoothness；`[0,1]` | diagnostic |
| [`novel_region_fidelity`](novel_region_fidelity.md) | `consistency_3d_4d` | `flow_dino` | `image_static`、`image_dynamic`、`video_static` | novel mask 内 PSNR/DINO 组合分 | 成功时 official；纯 image reference 通常为空 |
| [`reprojection_error`](reprojection_error.md) | `consistency_3d_4d` | `droid_slam` | `image_static`、`image_dynamic`、`video_static` | DROID reprojection error；`[0,1.0719]` 反向缩放 | 成功时 official |
| [`optical_flow_aepe`](optical_flow_aepe.md) | `consistency_3d_4d` | `sea_raft` | `image_static`、`image_dynamic`、`video_static` | prediction optical-flow AEPE 后端分数；`[0,1.192]` 反向缩放 | 成功时 official |
| [`depth_accuracy`](depth_accuracy.md) | `consistency_3d_4d` | `depth_anything` | formal 4 suite | paired-frame AbsRel；`[0,1]` 反向缩放 | diagnostic |
| [`depth_collision`](depth_collision.md) | `consistency_3d_4d` | `video_depth_anything` | formal 4 suite | 中心区域近距离碰撞 flag；`0/1` 反向缩放 | diagnostic |
| [`met3r_consistency`](met3r_consistency.md) | `consistency_3d_4d` | `met3r` | formal 4 suite | paired anchor MEt3R distance；`[0,2]` 反向缩放 | diagnostic |
| [`reconstruction_consistency`](reconstruction_consistency.md) | `consistency_3d_4d` | `world_r1_lpips` | formal 4 suite | World-R1 `S_recon`；`clip(1 - mean LPIPS, 0, 1)` | diagnostic |
| [`pmf`](pmf.md) | `physics` | `numpy_fft` | `image_static`、`image_dynamic`、`video_dynamic` | 预测/参考 3D FFT 能量分布相似度；`[0,6]` | 成功时 official；需要至少 2 帧 reference |
| [`motion_magnitude`](motion_magnitude.md) | `physics` | `sea_raft` | `image_static`、`image_dynamic`、`video_dynamic`、`experimental` | 光流幅度 z-score 映射到 `[0.25,0.75]` | 成功时 official |
| [`motion_smoothness`](motion_smoothness.md) | `physics` | `vfi_mamba` | `image_static`、`image_dynamic`、`video_dynamic`、`experimental` | `{MSE, SSIM, LPIPS}` 分量归一化后算术平均 | 成功时 official |
| [`long_sequence_dynamic_degree`](long_sequence.md) | `physics` | `long_sequence` | `image_static`、`image_dynamic`、`video_dynamic` | Long-Sequence dynamic degree；`[0,1]` | diagnostic |
| [`physical_fps_error`](temporal_calibration.md) | `real_time` | `pulse_of_motion` | `image_static`、`image_dynamic`、`video_dynamic` | `abs(avg_phyfps - meta_fps)`；`[0,30]` 反向缩放 | 成功时 official |
| [`physical_fps_pct_error`](temporal_calibration.md) | `real_time` | `pulse_of_motion` | `image_static`、`image_dynamic`、`video_dynamic` | FPS 相对误差；`[0,1]` 反向缩放 | 成功时 official |
| [`physical_fps_intra_video_cv`](temporal_calibration.md) | `real_time` | `pulse_of_motion` | `image_static`、`image_dynamic`、`video_dynamic` | 片段物理 FPS 变异系数；`[0,1]` 反向缩放 | 成功时 official |

## 独立 Physics Track

`config/benchmark_physics.yaml` 是 simulator-backed physics 配置，和默认 `config/benchmark.yaml` 的 `experimental` 不是同一个评测面。它携带 `mse`、`st_iou`、`depth_warp_charb`、`rgb_warp_charb`、`chamfer_4d`、`worldline_*`、`trajectory_*`、`final_state_accuracy` 等指标，并统一汇总到顶层 `physics` 维度。它使用同一套 report contract，但指标语义、artifact 需求和可比性都以 [Physics](dimensions/physics.md)、[Simulator-Backed Physics](physics/index.md) 与 [Physics Metrics & Contracts](physics/metrics.md) 为准。
