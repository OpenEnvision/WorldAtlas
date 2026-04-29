# Action Control

`action_control` 维度覆盖 scene control、camera control、object/action control。它关注模型是否遵循 prompt、相机指令、目标物体约束和显式运动监督。

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| prompt / scene control | [`prompt_alignment`](../prompt_alignment.md)、[`semantic_alignment`](../semantic_alignment.md) |
| camera control | [`camera_error`](../camera_error.md) |
| object control | [`object_detection`](../object_detection.md) |
| action / motion control | [`motion_accuracy`](../motion_accuracy.md) |

## 解读重点

- `prompt_alignment` 只在有显式 instruction supervision 且非 self-conditioned 时进入 official。
- `object_detection` 只对显式结构化 object annotation 打分，caption-derived object prompt 不计分。
- `camera_error` 与 `motion_accuracy` 都依赖强监督 sidecar；缺少 pose 或 dynamic mask 时不会被强行记 0。
