# Long Sequence（Memory）

`long_sequence` 维度覆盖状态记忆、外观保持与长时序 rollout 稳定性。它关注模型是否能在长视频中维持已见区域、物体存在性和回访行为，以及长时运动或画质表现，并将维度汇总结果写入 `per_dimension_summary.json`。核心问题是模型是否记得之前生成过什么。

### 覆盖范围

本维度按长时记忆的不同层面组织指标。回访稳定性由 [`memory_revisit_consistency`](../memory_revisit_consistency.md) 衡量，用于检查模型在重新访问场景时是否保持一致。

已见区域与物体持久性由 [`seen_region_preservation`](../seen_region_preservation.md) 和 [`object_permanence`](../object_permanence.md) 衡量，分别关注已见区域的外观保持与物体在长序列中的持续存在。Long-Sequence backend 还提供 [`long_sequence_motion_smoothness`](../long_sequence.md)，用于补充长时运动平滑性读数。

### 解读边界

这些指标主要回答“模型是否记得之前生成过什么”。`seen_region_preservation` 与 `object_permanence` 在有 reference video / mask 时更强；监督缺失时通常为 `not_applicable` 或非 official。bundled Long-Sequence 指标适合和 formal suite 指标一起看，不单独构成 headline 总分。
