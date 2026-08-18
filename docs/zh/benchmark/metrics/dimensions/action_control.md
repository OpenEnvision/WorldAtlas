# Action Control

**action_control** 维度评估生成 rollout 是否遵循 benchmark 声明的控制信号：文本指令、相机轨迹，以及显式运动监督。该维度将正式指标分数汇总写入 `per_dimension_summary.json`，是 WorldArena 的顶层评测轴之一。

这与 perceptual quality、temporal smoothness 或几何一致性不同。一条 prediction 可以看起来清晰流畅，却仍然违背所要求的场景内容、相机路径或规定运动。因此 action control 衡量的是**指令保真度**——画面变化是否与 evaluator 被要求执行的控制一致。

该维度覆盖三类控制信号。**Prompt 与 scene control** 由 [`prompt_alignment`](../prompt_alignment.md) 与 [`semantic_alignment`](../semantic_alignment.md) 承担；其中 `prompt_alignment` 在存在显式 instruction supervision 且样本非 self-conditioned 时，对 protocol anchor frames 与目标指令做对齐评分。**Camera control** 使用 [`camera_error`](../camera_error.md)，将预测相机运动与 pose sidecar 比较。**Action 与 motion control** 包括 mask 监督下的 [`motion_accuracy`](../motion_accuracy.md)，以及 keystroke 指标族——[`keystroke_strict_action_accuracy`](../keystroke_strict_action_accuracy.md)、[`keystroke_partial_action_accuracy`](../keystroke_partial_action_accuracy.md)、[`keystroke_traj_score`](../keystroke_traj_score.md)、[`keystroke_natet`](../keystroke_natet.md)、[`keystroke_nater`](../keystroke_nater.md) 与 [`keystroke_action_score`](../keystroke_action_score.md)。
