# Physics

`physics` 维度覆盖运动物理真实性与 simulator-backed physics 指标。它包含默认 benchmark 中的运动动态指标，也包含由 simulator world artifact 支撑的独立 physics track，并将维度汇总结果写入 `per_dimension_summary.json`。核心问题是生成视频中的运动与交互是否符合合理的物理动态。

### 评测轨道与文档组织

本维度由三条评测轨道共同构成：默认 formal suite 中的运动指标、simulator-backed physics，以及独立的 Physics Judge。当前页面说明三者如何共同归入 `physics` 维度；simulator-backed physics 的范围、配置、report namespace 与解释边界见 [Simulator-Backed Physics](../physics/index.md)，manifest 构建、预测目录、运行命令和报告产物见 [Physics Workflow](../physics/workflow.md)，指标契约、归一化、artifact 要求和 failure semantics 见 [Physics Metrics & Contracts](../physics/metrics.md)。

### 覆盖范围

默认 formal suite 的参考运动动态由 [`pmf`](../pmf.md) 衡量，运动幅度与平滑性由 [`motion_magnitude`](../motion_magnitude.md) 和 [`motion_smoothness`](../motion_smoothness.md) 衡量。Long-Sequence track 还提供 `long_sequence_dynamic_degree`，用于补充长时段运动动态读数。

Simulator-backed physics 提供一组由 simulator world artifact 支撑的原子指标，具体公式与 artifact 契约以 [Physics Metrics & Contracts](../physics/metrics.md) 为准。这组指标包括 `mse`、`st_iou`、`s_iou`、`ws_iou`、`abs_rel`、`rmse`、`delta1`、`delta2`、`delta3`、`depth_warp_l1`、`depth_warp_charb`、`rgb_warp_charb`、`epe`、`fl_all`、`one_px_out`、`chamfer_4d`、`worldline_l2_error`、`trajectory_rmse` 和 `final_state_accuracy`。

独立的 [Physics Judge](../physics/physics_judge.md) 也属于 `physics` 维度，但它不是 simulator-backed artifact group，而是由 `config/benchmark_physics_judge.yaml` 单独驱动的 VLM physics judge。

### 解读边界

默认 formal suite 中的 `pmf`、`motion_magnitude` 和 `motion_smoothness` 面向视频运动动态。`config/benchmark_physics.yaml` 是 simulator-backed physics 配置，不应和默认 `config/benchmark.yaml` 的 formal suite headline 直接混合。

simulator-backed 指标的数学公式、artifact 契约和 failure semantics 应以 [Simulator-Backed Physics](../physics/index.md)、[Physics Workflow](../physics/workflow.md) 和 [Physics Metrics & Contracts](../physics/metrics.md) 中的说明为准；Physics Judge 则应按照其独立配置和报告语义解读。
