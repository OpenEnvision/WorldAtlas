# 语义对齐

`semantic_alignment` 用来衡量预测结果是否接近参考视频的标注语义内容，而不只是接近 benchmark
prompt。

## 它测什么

- 预测内容与参考 caption 内容的语义接近度
- 相对参考内容的语义一致，而不是 prompt following
- 一个诊断性代理信号，用来看“这个视频像不像同一类内容”

## 当前算法

1. WorldArena 从 `sample.annotation_path` 下的 `caption.json` 组装参考文本；如果存在指令标签，
   会把它们追加进去作为补充上下文。
2. 如果标注文本缺失，则回退到 `sample.prompt_current` 或 `sample.prompt_target`。
3. 使用 `openai/clip-vit-base-patch16` 的图像编码器对预测采样帧做编码。
4. 使用同一个 CLIP 的文本编码器对参考文本做编码。
5. 对每一帧计算图像特征和参考文本特征的 cosine，相似度再用 `(cos + 1) / 2` 映射到 `[0, 1]`。
6. 原始分数是所有采样帧分数的平均值。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 预测的采样帧 |
| 参考输入 | 参考标注文本，不直接读取参考像素 |
| 标注输入 | `caption.json`、可选 `instructions.json`、回退 prompt |
| 原始输出 | `[0, 1]` 内的单个平均语义分数 |
| 归一化输出 | 默认配置下与原始值基本相同 |
| 细节字段 | `reference_text_source`、`reference_text`、`frame_scores`、帧索引元数据 |
| 失败方式 | 若 CLIP 不可用或参考文本无法构造，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 集成状态

| 项目 | 结论 |
| --- | --- |
| 当前聚合 | 仅 `diagnostic` |
| 论文可用性 | `仅作辅助` |

- 它的价值：和 `prompt_alignment` 不同，它锚定的是参考侧内容。
- 它的限制：仍然依赖标注文本质量，也仍然是 CLIP 语义代理。
- 建议用法：和 prompt-following、质量指标一起看，用来区分“没跟 prompt”还是“没接近参考内容”。
