# Object Detection

**object_detection** 用来衡量在 benchmark 规定的 prompt-matching 合同下，预测帧里是否能恢复出要求的场景物体。

## 这个指标测什么

- 生成帧里要求物体的可恢复性
- 在开放词汇检测器下的物体存在性
- 它不是 box 精度指标，不是完整场景理解指标，也不是物体运动正确性指标

## 具体算法

1. WorldArena 会从标注目录导出 `prompt_candidates`。
2. prompt 只接受标注 payload 里的显式结构化 object 字段。
   Caption 文本里提取出的 noun phrase 只用于数据审计，不用于该指标打分。
3. 指标构造 `prompt_string = "scene, " + ", ".join(prompt_candidates)`。
4. 被包装的第三方指标会在每一帧解码图像上运行 `Grounding DINO`，阈值为 `box_threshold = 0.4`、`text_threshold = 0.4`。
5. `Grounding DINO` 会输出检测到区域对应的短语。
6. 匹配分两轮：
    - 先做 exact phrase match
    - 再做基于 spaCy 轻量词性的 noun overlap partial match
7. 单帧分数为 `frame_score = min(matched_prompt_count / len(prompt_candidates), 1.0)`。
8. 原始指标值为所有帧 `frame_score` 的平均值，即 `raw = mean frame_score`。
9. 当前归一化是直接的 `[0, 1]` 有界缩放。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 无 |
| 标注输入 | 标注目录里的结构化 object 标签 |
| 原始输出 | `[0, 1]` 上的单个标量 object recoverability 分数 |
| 归一化输出 | 同一个 `[0, 1]` 标量 |
| 细节字段 | `prompt_source`、`prompt_candidates`、`prompt_string`、`frame_count`、`native_score` |
| 失败形式 | 如果缺少显式 object prompt，则返回 `not_applicable`；如果 detector runtime 失败，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 只有 prompt 来自显式标注时才计算并进入 `official`；caption 派生 prompt 不打分，记为 `not_applicable` |
| 论文可用性结论 | `仅作辅助` |

- 它不够更强的原因在于：这里没有 box 级真值对比。这个指标只是在问，开放词汇检测器能否从生成帧里恢复出要求的物体短语。
- 匹配阶段本身就是启发式的，混合了 exact string matching 和 noun-overlap matching；有分析价值，但不等价于严格 detection benchmark。
- caption 派生 prompt 已被排除在打分路径之外，因为它不能支撑正式 object-control 证据。
- 建议用法：如果 `prompt_source = annotation_explicit`，它可以作为辅助性的 object recoverability 指标；但不要把它包装成严格的检测指标。
