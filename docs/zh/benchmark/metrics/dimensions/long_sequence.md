# Long Sequence（Memory）

`long_sequence` 维度覆盖状态记忆、外观保持与长时序 rollout 稳定性。它关注模型是否能在长视频中维持主体、背景、已见区域、物体存在性，以及闭环回访时的场景记忆。

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| anchor 记忆与回访稳定性 | [`anchor_consistency`](../anchor_consistency.md)、[`memory_revisit_consistency`](../memory_revisit_consistency.md) |
| 预测内部长时一致性 | [`subject_consistency`](../subject_consistency.md)、[`background_consistency`](../background_consistency.md) |
| 已见区域与物体持久性 | [`seen_region_preservation`](../seen_region_preservation.md)、[`object_permanence`](../object_permanence.md) |
| 闭环回访 | [`memory_revisit_consistency`](../memory_revisit_consistency.md) |
| Long-Sequence backend 诊断 | [`long_sequence_subject_consistency`](../long_sequence.md)、[`long_sequence_background_consistency`](../long_sequence.md)、[`long_sequence_motion_smoothness`](../long_sequence.md) |

## 解读重点

- 这些指标主要回答“模型是否记得之前生成过什么”。
- `seen_region_preservation` 与 `object_permanence` 在有 reference video / mask 时更强；监督缺失时通常为 `not_applicable` 或 diagnostic。
- bundled Long-Sequence 指标是诊断维度，适合和 formal suite 指标一起看，不单独构成 headline 总分。
