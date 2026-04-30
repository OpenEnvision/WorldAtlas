# Long Sequence（Memory）

`long_sequence` 维度覆盖状态记忆、外观保持与长时序 rollout 稳定性。它关注模型是否能在长视频中维持已见区域、物体存在性、回访行为，以及长时运动或画质表现。

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| 回访稳定性 | [`memory_revisit_consistency`](../memory_revisit_consistency.md) |
| 已见区域与物体持久性 | [`seen_region_preservation`](../seen_region_preservation.md)、[`object_permanence`](../object_permanence.md) |
| Long-Sequence backend 诊断 | [`long_sequence_motion_smoothness`](../long_sequence.md) |

## 解读重点

- 这些指标主要回答“模型是否记得之前生成过什么”。
- `seen_region_preservation` 与 `object_permanence` 在有 reference video / mask 时更强；监督缺失时通常为 `not_applicable` 或 diagnostic。
- bundled Long-Sequence 指标是诊断维度，适合和 formal suite 指标一起看，不单独构成 headline 总分。
