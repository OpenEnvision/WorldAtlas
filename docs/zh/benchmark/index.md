# 基准评测总览

WorldArena 为可控 world generation 报告按 suite 划分的指标向量。当前默认配置声明 `32` 个去重指标；benchmark 契约保持收敛：固定 manifest 结构、固定评测协议、显式样本级资格规则，并且不定义 benchmark 级总分。

仓库级架构比当前正式 benchmark 面更广。模型配置和 manifest 已经显式携带任务元数据，因此同一套框架未来可以承载 3D、4D、重建和 simulator-backed world-model 评测，而不需要第二套配置系统。参见 [架构](../architecture.md)。

!!! note "数据统计在别处"
    集合规模、标注覆盖率、caption 分布和媒体统计属于 [数据分析](../overview.md) 部分，不会在 Benchmark 章节中重复维护。

## Suite 指标向量

| Suite | 模态 | 条件输入契约 | 满足资格时进入官方聚合的指标 | 诊断指标与备注 |
| --- | --- | --- | --- | --- |
| `image_static` | image -> video | `reference_image`，1 帧；prediction 是 I2V rollout | video-output 指标并集。常见 official：`prompt_alignment`*、`object_detection`*、`style_consistency`、`perceptual_quality`、`reprojection_error`、`optical_flow_aepe`、`motion_magnitude`、`motion_smoothness`、`physical_fps_*`；有 pose/mask/reference-video sidecar 时还会启用对应条件指标 | `semantic_alignment`、`brightness_consistency`、`color_temperature_consistency`、`sharpness_retention`、`depth_accuracy`、`depth_collision`、`psnr`、`ssim`、`met3r_consistency`、`reconstruction_consistency`、`long_sequence_*`；依赖 reference video 的指标在纯 image reference 上通常为空 |
| `image_dynamic` | image -> video | `reference_image`，1 帧；prediction 是 I2V rollout | 同 `image_static`，但 motion/object 相关监督通常更有机会覆盖 | 同 `image_static` |
| `video_static` | video | `first_frame`，1 帧 | `prompt_alignment`*、`style_consistency`、`perceptual_quality`、`camera_error`、`seen_region_preservation`、`novel_region_fidelity`、`object_detection`*、`reprojection_error`、`optical_flow_aepe` | `semantic_alignment`、`brightness_consistency`、`color_temperature_consistency`、`sharpness_retention`、`depth_accuracy`、`depth_collision`、`psnr`、`ssim`、`met3r_consistency`、`reconstruction_consistency` |
| `video_dynamic` | video | `reference_video`，prefix continuation | `prompt_alignment`*、`style_consistency`、`perceptual_quality`、`object_permanence`、`pmf`、`motion_accuracy`、`motion_magnitude`、`motion_smoothness`、`physical_fps_error`、`physical_fps_pct_error`、`physical_fps_intra_video_cv` | `semantic_alignment`、`brightness_consistency`、`color_temperature_consistency`、`sharpness_retention`、`depth_accuracy`、`depth_collision`、`psnr`、`ssim`、`met3r_consistency`、`reconstruction_consistency`、`long_sequence_*` |
| `experimental` | mixed | mixed | `motion_magnitude`、`motion_smoothness`（仅显式评估该 suite；非 formal headline） | `memory_revisit_consistency` |

\* 仍然要通过样本级 eligibility gate。`prompt_alignment` 只在显式 instruction 且非 self-conditioned 时计算；`object_detection` 只在 object prompt 来自显式结构化标注时计算。其他情况记为 `not_applicable`，不产生分数。详见 [协议与资格](protocol.md)。

`video_dynamic` 使用 benchmark 的 prefix continuation 契约：默认把参考 rollout 的前 `25%` 视作条件输入，并且至少保留 `4` 帧；其后的 continuation 区域才是主要评测目标。

## 六个评测支柱

WorldArena 现在把 benchmark 指标归入六个有意保持宽口径的评测支柱。它们应该被理解为 benchmark 覆盖能力面的能力级分桶，而不是按后端或单一实现细分出来的窄类别：

| 维度键 | 展示名称 | 覆盖范围 |
| --- | --- | --- |
| `long_sequence` | Long sequence（memory） | 状态记忆、外观保持与长时序 rollout 稳定性 |
| `action_control` | Action control | scene control、camera control、object/action control |
| `consistency_3d_4d` | 3D/4D consistency | 几何、多视角、重投影与运动场一致性 |
| `physics` | Physics | 运动物理真实性与 simulator-backed physics 指标 |
| `quality` | Quality | 风格、感知质量、低层外观稳定性与重建式保真 |
| `real_time` | Real time | 节奏与 FPS 标定 |

`per_dimension_summary.json` 会基于这六个评测支柱输出官方 suite-level 汇总。详见 [指标总览](metrics/index.md)。

## Headline 报告契约

- 主结果是 `per_metric_summary.json`：按 suite 汇总的、`eligibility_status = official` 的数值指标平均值。
- `per_dimension_summary.json` 会把同一批 official 指标进一步汇总成六个顶层维度。
- `leaderboard.csv` 只是同一组官方 suite-by-metric 向量的扁平导出。
- `diagnostic_metric_summary.json` 会单独保存。它只收集设计上明确为诊断的分数。
- `not_applicable_summary.json` 会记录因强监督不足而跳过的指标；这些条目既不是失败，也不是可解释分数。
- 某个指标被配置出来，并不意味着它自动成为 headline metric。它既要成功运行，也要满足该样本对应的资格门槛。
- WorldArena 不定义 benchmark 级标量，也不定义官方单分 suite score。

## Physics 维度中的 simulator-backed 配置

`config/benchmark_physics.yaml` 定义了一套 simulator-backed physics 配置，track 名为 `physics_i2v`。它同样复用了 suite 名称 `experimental`，但这和默认 `config/benchmark.yaml` 中那个仅用于诊断的 `experimental` 切片不是一回事。

- physics run 仍然会输出自己的 `per_metric_summary.json`、`leaderboard.csv`、`per_sample_scores.jsonl` 和 failure report
- 其中汇总的是 simulator-backed 指标，例如 `mse`、`st_iou`、`depth_warp_charb`、`chamfer_4d`、`worldline_*`
- 这些结果位于独立的 physics namespace 中，不应和上面四个 formal suite 的公开向量混合解释

详见 Metrics 下的 [Physics](metrics/dimensions/physics.md) 与 [Simulator-Backed Physics](metrics/physics/index.md)。

## 文档地图

- [协议与资格](protocol.md)：采样、条件输入、归一化以及 official / diagnostic 规则。
- [工作流与输出](workflow.md)：命令、预测目录布局、运行时解析、报告文件，以及 physics manifest 工作流。
- [Metrics / Physics](metrics/dimensions/physics.md)：运动物理真实性与 simulator-backed physics 指标的统一入口。
- [指标总览](metrics/index.md)：`config/benchmark.yaml` 当前声明的完整指标面，按维度和 suite 归类。
- [运行时布局](../thirdparty.md)：缓存布局、第三方 checkout 和 checkpoint 集成方式。
