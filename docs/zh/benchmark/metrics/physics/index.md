# Simulator-Backed Physics

WorldArena 把 simulator-backed physics 指标放在 `physics` 维度下，因为它比较的是生成视频与 simulator 导出的世界状态，而不只是渲染出来的参考媒体。physics track 复用了同一套 manifest、report 和 CLI 基础设施，但它使用独立配置、manifest、报告目录和指标契约。

放到更宽的六支柱 benchmark 叙事里看，这整条 track 应被理解为 `physics` 支柱的高保真端，而不是另一套独立的概念 taxonomy。

!!! note "不要把两个 experimental suite 混为一谈"
    `config/benchmark.yaml` 里有一个仅作补充的 `experimental` suite。`config/benchmark_physics.yaml` 也使用了 suite 名 `experimental`，但这里指的是 simulator-backed physics track，并且它的指标会进入该 physics run 自己的 `per_metric_summary.json`。

## 配置身份

| 项目 | 当前值 |
| --- | --- |
| 配置 | `config/benchmark_physics.yaml` |
| Suite | `experimental` |
| Track | `physics_i2v` |
| 模态 | video |
| 条件输入策略 | `external_image` |
| 指标后端 | 所有已配置指标都走 `simulator_reference` |
| 官方协议 | `segment_random` |
| 兼容协议 | `anchor_compat` |
| 目标 FPS | `24` |
| 配置中的产物根目录 | `artifacts/benchmark/physics/inventory`、`artifacts/benchmark/physics/manifest`、`artifacts/benchmark/physics/reports`、`.cache/worldarena/benchmark_physics` |

这条 track 仍然通过标准的 `worldarena evaluate` 命令来评测。真正变化的是 manifest 内容：每个样本都会带上 simulator 专用的 `physics_spec` 和 `physics_case`，让 evaluator 能定位到导出的 `world_gt` 产物以及外部 physics pipeline checkout。

## 样本来源

WorldArena 当前支持三种 physics 样本构建方式：

1. 通过 `build-physics-manifest` 从 simulator pipeline 导出的资产根目录构建。
   它要求存在 `assets/<sample_id>.mp4` 和 `assets/<sample_id>.json`，必要时会导出首帧作为条件图像，样本 id 形如 `physics__<source_sample_id>__i2v`。
2. 通过 `build-physics-seed-manifest` 从 seed image manifest 加可选 case catalog 构建。
   它会生成 image-conditioned continuation case，样本 id 形如 `physics_seed__<case_id>__i2v`。
3. 通过 `build_online_physics_sample(...)` 和 `evaluate_online_physics_i2v(...)` 这样的 Python helper 在线构建。
   这条路径适合 batch CLI 之外的单样本 i2v 调试。

## Manifest 负载

physics 样本会在标准 benchmark schema 之上新增两个关键元数据块。

`physics_spec` 描述生成视频应该满足的契约：

- benchmark family 和 backend 名称
- source sample id 与 prompt 元数据
- 按 `video_physics`、`geometry_3d`、`temporal_consistency`、`world_4d`、`trajectory_dynamics` 分组的 `metrics_design`
- 对应的 `optional_metrics_design`，用于声明 backend 能识别但默认未启用的输出
- `metric_contract`，例如 `required_core_metrics`、`require_full_frame_coverage`、`require_pred_world_artifacts`
- 从源元数据复制过来的 case taxonomy 与 simulator hints

`physics_case` 描述运行时产物的位置：

- `physics_pipeline_root` 与解析后的 `physics_data_root`
- 本地 case root、simulator root 和 `submitted_video_path`
- `metadata_template_path` 以及内嵌的 metadata template 负载
- 本地 `world_gt_root`
- 外部 simulator 脚本使用的 export benchmark 名称与 render split

## 报告行为

physics run 复用和主 benchmark 相同的输出文件契约：

- `run_manifest.json`
- `per_metric_summary.json`
- `per_dimension_summary.json`
- `per_sample_scores.jsonl`
- `metric_failures.json`
- `leaderboard.csv`

但有几个关键差异：

- 报告中的 suite 是 `experimental`，而不是四个公开 formal suite 之一
- physics 指标在成功返回数值时默认使用 `eligibility_status = official`，因此 `per_metric_summary.json` 就是 physics run 的主汇总
- `per_dimension_summary.json` 现在会把启用的 simulator-backed 指标再汇总成单一的顶层 `physics` 维度
- 这些数值不应与 `config/benchmark.yaml` 下四个 formal suite 的公开向量混合或直接比较

## 继续阅读

- [Physics Workflow](workflow.md)：外部 checkout 要求、manifest 构建命令、评测生命周期与 readiness 状态
- [Physics Metrics & Contracts](metrics.md)：指标分组、归一化范围、必需 artifact 与成功条件
- [工作流与输出](../../workflow.md)：physics track 与公开 benchmark 共用的报告布局
