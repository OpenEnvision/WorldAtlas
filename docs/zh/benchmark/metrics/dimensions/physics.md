# Physics

`physics` 维度覆盖运动物理真实性与 simulator-backed physics 指标。它包含默认 benchmark 中的运动动态指标，也包含独立 physics track 中由 simulator world artifact 支撑的指标。

## 总览

| 页面 | 用途 |
| --- | --- |
| [Physics](physics.md) | 当前维度入口：说明 public benchmark 运动指标、simulator-backed physics、Physics Judge 三者如何共同归入 `physics` 维度 |
| [Simulator-Backed Physics](../physics/index.md) | simulator-backed physics track 的范围、配置、report namespace 与解释边界 |
| [Physics Workflow](../physics/workflow.md) | manifest 构建、预测目录、运行命令和报告产物 |
| [Physics Metrics & Contracts](../physics/metrics.md) | 指标契约、归一化、artifact 要求和 failure semantics |

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| 参考运动动态 | [`pmf`](../pmf.md) |
| 运动幅度与平滑 | [`motion_magnitude`](../motion_magnitude.md)、[`motion_smoothness`](../motion_smoothness.md) |
| Long-Sequence 动态诊断 | [`long_sequence_dynamic_degree`](../long_sequence.md) |
| Simulator Based Physics evaluation 指标分组 | [Video Physics](../physics/video_physics.md)、[Geometry 3D](../physics/geometry_3d.md)、[Temporal Consistency](../physics/temporal_consistency.md)、[World 4D](../physics/world_4d.md)、[Trajectory Dynamics](../physics/trajectory_dynamics.md)、[Final State](../physics/final_state.md) |
| 独立 VLM physics judge | [Physics Judge](../physics/physics_judge.md) |

## 解读重点

- 默认 formal suite 中的 `pmf`、`motion_magnitude`、`motion_smoothness` 面向视频运动动态。
- `config/benchmark_physics.yaml` 是 simulator-backed physics 配置，不应和默认 `config/benchmark.yaml` 的 formal suite headline 直接混合。
- simulator-backed 指标的数学公式、artifact 契约和 failure semantics 以总览中的 [Simulator-Backed Physics](../physics/index.md)、[Physics Workflow](../physics/workflow.md) 和 [Physics Metrics & Contracts](../physics/metrics.md) 为准。
- [Physics Judge](../physics/physics_judge.md) 属于 `physics` 维度，但它不是 simulator-backed artifact group；它由 `config/benchmark_physics_judge.yaml` 单独驱动。
