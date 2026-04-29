# 已见区域保真

`seen_region_preservation` 问的是：那些在参考视频第一帧里已经出现、并且后续仍然可见的场景区域，预测是否把它们保留下来了。

## 这个指标测什么

- 第一帧里已经可见、后续仍然存在的区域保真度
- 只在这些持续可见区域上做 prediction-vs-reference 的 masked 对比
- 不是 prompt following，也不是 camera ground truth accuracy，更不是动态前景跟踪指标

## 具体算法

1. 该指标要求参考视频至少有两帧采样结果。
2. WorldArena 把第一帧参考帧当作 source frame。
3. 它会尝试从 `sample.mask_path` 或 `annotation_path/dyn_masks.npz` 里读取 ignore mask，避免动态前景主导这个静态场景指标。
4. 对每个后续参考帧，计算 source frame 到 target frame 的前向 / 后向 Farneback 光流。
5. 一个像素会被视为“已见区域”，需要同时满足：
    - backward-warp 后仍然落在图像内部
    - 前后向光流一致性误差低于当前阈值
    - 没有被可选的 ignore mask 去掉
6. 得到的 seen mask 会先做轻量形态学清理。
7. 对每个后续时刻，把 prediction resize 到 reference 分辨率，并且只在 seen mask 内比较 prediction 和 reference。
8. 每帧分数是下列可用分量的平均值：
    - 映射到 `[0, 1]` 的 masked PSNR
    - 基于 masked crop 的 DINOv2 相似度
9. 帧分数按 mask 覆盖率加权，最后取加权平均作为原始分数。
10. 当前默认归一化就是 `[0, 1]` 上的直接有界缩放。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 采样后的 prediction 视频帧 |
| 参考输入 | 采样后的 reference 视频帧 |
| 标注输入 | 来自 `mask_path` 或 `dyn_masks.npz` 的可选 ignore mask |
| 原始输出 | 一个 `[0, 1]` 范围内的加权平均标量 |
| 归一化输出 | 默认配置下与原始值相同的有界标量 |
| 细节字段 | `frame_scores`，其中包含 `frame_index`、`coverage`、`psnr_component`、`dino_component`、`score`，以及帧索引元数据 |
| 失败形式 | 如果无法形成有效的 seen-region 分数，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 集成说明

- 当前聚合状态：在 `image_static`、`image_dynamic`、`video_static` 中，只要成功产出数值就属于 `official`；纯 image reference 通常会因为没有多帧 reference 而为空。
- 很小或空的 mask 会被跳过，而不是被当作 0 分。
- 这个指标最好和 [`novel_region_fidelity`](novel_region_fidelity.md) 一起看，因为两者分别回答“保持住原来可见的部分”和“新露出来的部分画得对不对”。
