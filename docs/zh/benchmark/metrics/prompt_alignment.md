# Prompt Alignment

`prompt_alignment` 用来衡量预测视频在 benchmark 协议锚点帧上是否匹配 `sample.prompt_target`。

## 这个指标测什么

- 生成内容与目标指令之间的语义一致性
- 锚点帧层面的指令跟随，而不是整段视频的长时序理解
- 测的是内容正确性，不是 realism、smoothness，也不是对参考结果的复制程度

## 具体算法

1. 对预测视频先均匀下采样到最多 `benchmark.motion_frame_count` 帧，然后当前 official 协议 `segment_random` 会在这些采样帧里每个 segment 选一个协议帧。image suite 的参考输入可以是单帧图像，但模型输出仍必须是视频。
2. 对每个预测锚点帧 `f_i`，WorldArena 调用 `torchmetrics.multimodal.CLIPScore`，模型为 `openai/clip-vit-base-patch16`，文本输入为 `sample.prompt_target`。
3. 原始 CLIPScore `s_i` 会被转换成有界帧分数：`frame_score_i = clamp(s_i / 100, 0, 1)`。
4. 原始指标值为 `raw = mean_i frame_score_i`。
5. 当前配置里 benchmark 归一化是恒等映射，因为 `lower = 0`、`upper = 1`、`higher_is_better = true`，所以 `normalized = raw`。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 只接受 video prediction；只使用预测视频锚点帧 |
| 参考输入 | 无 |
| 标注输入 | `sample.prompt_target`、`sample.has_instruction`、`sample.conditioning_strategy` |
| 原始输出 | `[0, 1]` 上的单个标量平均锚点分数 |
| 归一化输出 | 同一个 `[0, 1]` 标量 |
| 细节字段 | `frame_scores`、`native_frame_scores`、`native_score_range`、`frame_indices`、`sampled_frame_indices` |
| 失败形式 | 如果 prediction 不是视频，或 CLIPScore backend 不可用，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 只有当 `sample.has_instruction = true` 且 `sample.conditioning_strategy != "self"` 时才计算并进入 `official`；否则是 `not_applicable` |
| 论文可用性结论 | `主表可用` |

- 相对更强的原因：算法直接在本仓库实现，分数变换简单，现有测试也覆盖了缩放与 eligibility 路由。
- prompt 文本必须描述目标视频结果；image suite 的 `prompt_target` 也应写成 video rollout 指令，而不是 image generation 指令。
- 缺少显式 instruction 或 self-conditioned 的样本不再给分，因为这类分数无法支撑严格的 prompt-following 结论。
- 仍然不能单独使用的原因：CLIPScore 本质上仍是语义 proxy，不是人类判断；同时锚点帧评估也可能漏掉长时序错误。
- 建议用法：可以作为主表里的语义指标，但必须搭配质量类和控制类指标一起看。
