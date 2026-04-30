# Physics Metrics & Contracts

`config/benchmark_physics.yaml` 中的 simulator-backed 指标当前都解析到 `simulator_reference` backend。benchmark runner 会对每个样本只执行一次底层 physics evaluator，并把结果缓存下来，因此同一个预测文件上启用多个 physics 指标不会重复运行外部 evaluator。

从报告维度上看，physics track 中的 configured / optional simulator-backed 指标，以及 `config/benchmark_physics_judge.yaml` 中的 judge 指标，都归入顶层 `physics` 维度。

## Simulator Based Physics evaluation 指标分组

`physics_spec.metrics_design` 会把必需指标分成五个 core simulator-backed group，另加一个场景条件 final-state group。每个 group 的详细输入、计算语义、方向和失败模式拆在单独页面里：

| simulator-backed group | 指标 | 详细页面 |
| --- | --- | --- |
| `video_physics` | `mse`、`st_iou`、`s_iou`、`ws_iou` | [Video Physics](video_physics.md) |
| `geometry_3d` | `abs_rel`、`rmse`、`delta1`、`delta2`、`delta3` | [Geometry 3D](geometry_3d.md) |
| `temporal_consistency` | `depth_warp_l1`、`depth_warp_charb`、`rgb_warp_charb`、`epe`、`fl_all`、`one_px_out`，以及可选 `rgb_warp_lpips` | [Temporal Consistency](temporal_consistency.md) |
| `world_4d` | `chamfer_4d`、`worldline_l2_error`、`worldline_mean_drift`、`worldline_final_drift`、`worldline_fail_rate`、`worldline_length`，以及可选 `novel_time_*` | [World 4D](world_4d.md) |
| `trajectory_dynamics` | `trajectory_rmse`、`final_position_error`、`speed_similarity`、`acceleration_similarity`、`directional_consistency` | [Trajectory Dynamics](trajectory_dynamics.md) |
| `final_state` | 样本声明 final-state 规则或标签时启用的 optional `final_state_accuracy` | [Final State](final_state.md) |

这些分组会被展平成每个 physics 样本中携带的 `required_core_metrics` 契约。自定义配置如果启用了 optional metric，也要同步声明 backend 和 normalization。

## Physics Judge

VLM judge 版本的 physics 评测由单独配置驱动。它属于顶层 `physics` 维度，但不属于上面的 simulator-backed artifact group：

| 指标族 | 指标 | 详细页面 |
| --- | --- | --- |
| `physics_judge` | `physics_judge_overall`、`physics_judge_phenomenon`、`physics_judge_causality`、`physics_judge_material`、`physics_judge_temporal`、`physics_judge_prompt` | [Physics Judge](physics_judge.md) |

## 归一化映射

当前配置声明的归一化范围如下。逐项指标解释见上面的 group 页面。

WorldArena 的通用归一化公式是：

\[
\mathrm{norm}(x)=
\begin{cases}
\mathrm{clip}_{[0,1]}\left(\dfrac{x-l}{u-l}\right), & \text{higher\_is\_better=true}\\[8pt]
\mathrm{clip}_{[0,1]}\left(\dfrac{u-x}{u-l}\right), & \text{higher\_is\_better=false}
\end{cases}
\]

其中 \(x\) 是 raw metric，\(l\) 是 `lower`，\(u\) 是 `upper`。若 \(u\approx l\)，实现直接对 raw value 做 \([0,1]\) clipping。

| 指标 | 来源 | 更优方向 | 映射到 `[0, 1]` 的原始区间 |
| --- | --- | --- | --- |
| `mse` | `video_physics` | 越低越好 | `[0.0, 0.25]` |
| `st_iou` | `video_physics` | 越高越好 | `[0.0, 1.0]` |
| `s_iou` | `video_physics` | 越高越好 | `[0.0, 1.0]` |
| `ws_iou` | `video_physics` | 越高越好 | `[0.0, 1.0]` |
| `abs_rel` | `geometry_3d` | 越低越好 | `[0.0, 1.0]` |
| `rmse` | `geometry_3d` | 越低越好 | `[0.0, 5.0]` |
| `delta1` | `geometry_3d` | 越高越好 | `[0.0, 1.0]` |
| `delta2` | `geometry_3d` | 越高越好 | `[0.0, 1.0]` |
| `delta3` | `geometry_3d` | 越高越好 | `[0.0, 1.0]` |
| `depth_warp_l1` | `temporal_consistency` | 越低越好 | `[0.0, 1.0]` |
| `depth_warp_charb` | `temporal_consistency` | 越低越好 | `[0.0, 1.0]` |
| `rgb_warp_charb` | `temporal_consistency` | 越低越好 | `[0.0, 0.5]` |
| `epe` | `temporal_consistency` | 越低越好 | `[0.0, 10.0]` |
| `fl_all` | `temporal_consistency` | 越低越好 | `[0.0, 1.0]` |
| `one_px_out` | `temporal_consistency` | 越低越好 | `[0.0, 1.0]` |
| `chamfer_4d` | `world_4d` | 越低越好 | `[0.0, 1.0]` |
| `worldline_l2_error` | `world_4d` | 越低越好 | `[0.0, 1.0]` |
| `worldline_mean_drift` | `world_4d` | 越低越好 | `[0.0, 1.0]` |
| `worldline_final_drift` | `world_4d` | 越低越好 | `[0.0, 1.0]` |
| `worldline_fail_rate` | `world_4d` | 越低越好 | `[0.0, 1.0]` |
| `worldline_length` | `world_4d` | 越高越好 | `[0.0, 1.0]` |
| `trajectory_rmse` | `trajectory_dynamics` | 越低越好 | `[0.0, 1.0]` |
| `final_position_error` | `trajectory_dynamics` | 越低越好 | `[0.0, 1.0]` |
| `speed_similarity` | `trajectory_dynamics` | 越高越好 | `[-1.0, 1.0]` |
| `acceleration_similarity` | `trajectory_dynamics` | 越高越好 | `[-1.0, 1.0]` |
| `directional_consistency` | `trajectory_dynamics` | 越高越好 | `[0.0, 1.0]` |
| `final_state_accuracy` | `final_state` | 越高越好 | `[0.0, 1.0]` |
| `physics_judge_*` | `physics_judge` | 越高越好 | `[0.0, 1.0]` |

## Artifact 契约

physics 评测会区分本地 ground-truth 导出和 predicted world 导出。

### `world_gt/`

| 契约层级 | 文件 |
| --- | --- |
| 最低视频就绪条件 | `rgb.mp4` |
| 完整 core-metric 就绪条件 | `rgb.mp4`、`depth.npz`、`flow.npz`、`mask.npz`、`occlusion.npz`、`cameras.npz`、`points4d.npz`、`worldlines.npz` |
| 可选附加项 | `novel_time_depth.npz`、`novel_time_rgb.npz`、`rgb_frames/`、`export_info.json` |

### `pred_world/`

| 契约层级 | 文件 |
| --- | --- |
| 当前 core contract 的必需项 | `depth.npz`、`mask.npz`、`flow.npz`、`cameras.npz` |
| 可选附加项 | `occlusion.npz`、`points4d.npz`、`worldlines.npz`、`novel_time_depth.npz`、`novel_time_rgb.npz` |

对于 `trajectory_dynamics`，若 `pred_world/worldlines.npz` 已存在，WorldArena 会直接复用；若缺失，则会先用 `pred_world/depth.npz`、`pred_world/cameras.npz` 和 `world_gt/worldlines.npz` 中的 pixel tracks 重建 `pred_world/worldlines.npz`，再调用 evaluator。

当前样本契约会设置：

- `require_full_frame_coverage = true`
- `require_pred_world_artifacts = true`

## 成功与失败语义

`evaluate_generated_video(...)` 只有在以下条件全部满足时才会把样本标记为 `success = true`：

- `inspect_physics_case(...)` 最终得到 `status = ready`
- 结果里包含 `required_core_metrics` 中声明的全部指标
- frame count 与 frame coverage 满足 metric contract
- 当 `require_pred_world_artifacts = true` 时，`pred_world` 生成成功

常见 contract failure 包括：

- `prediction frame count does not match the reference rollout`
- `prediction covers only ... of the reference rollout`
- `missing required core physics metrics: ...`
- `predicted world artifacts are required for this physics evaluation`

这些细节会在 `per_sample_scores.jsonl` 中暴露为：

- `report_path`
- `world_gt_root`
- `physics_status`
- `pred_world_status`
- `world_gt_status`
- `metric_contract`
- `missing_world_artifacts`
- 当 evaluator 漏掉某个必需指标时，还会附带 `missing_required_metrics`
