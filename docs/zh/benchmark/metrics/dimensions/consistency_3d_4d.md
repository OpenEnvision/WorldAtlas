# 3D/4D Consistency

`consistency_3d_4d` 维度覆盖几何、多视角、重投影与运动场一致性。它关注生成结果是否能支持稳定的三维/四维解释，而不只是单帧视觉质量。

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| 新显露区域 | [`novel_region_fidelity`](../novel_region_fidelity.md) |
| 多视角 / 3D 表征一致性 | [`met3r_consistency`](../met3r_consistency.md)、[`reconstruction_consistency`](../reconstruction_consistency.md) |
| 相机与重投影 | [`reprojection_error`](../reprojection_error.md) |
| 深度 | [`depth_accuracy`](../depth_accuracy.md)、[`depth_collision`](../depth_collision.md) |
| 运动场 | [`optical_flow_aepe`](../optical_flow_aepe.md) |

## 解读重点

- 这些指标更接近“生成视频背后的几何是否自洽”。
- `reprojection_error`、`optical_flow_aepe` 成功时通常可进入 official；`depth_accuracy`、`depth_collision`、`met3r_consistency`、`reconstruction_consistency` 当前更多作为 diagnostic。
- 这组指标不直接衡量 prompt following 或 aesthetic quality，需要和 `action_control`、`quality` 一起报告。
