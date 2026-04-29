# 新显露区域保真

`novel_region_fidelity` 问的是：当参考相机视角暴露出第一帧中看不到的新区域时，预测是否把这些新显露区域合理地画出来了。

## 这个指标测什么

- 随时间推移 newly visible 区域的保真度
- 只在这些新暴露区域上做 prediction-vs-reference 的 masked 对比
- 不是直接的 camera-control 指标，也不是纯粹的 hallucination 检测器

## 具体算法

1. 该指标要求参考视频至少有两帧采样结果。
2. WorldArena 先通过第一帧参考图与后续参考图之间的前后向 Farneback 光流，推导出“已见区域”。
3. 然后把膨胀后的 seen mask 取补集，得到 “novel region”，并且可选地去掉来自 `mask_path` 或 `dyn_masks.npz` 的动态前景区域。
4. 对每个后续时刻，把 prediction resize 到 reference 分辨率，并且只在 novel mask 内比较 prediction 和 reference。
5. 每帧分数是下列可用分量的平均值：
    - 映射到 `[0, 1]` 的 masked PSNR
    - 基于 masked crop 的 DINOv2 相似度
6. 帧分数按 novel 区域覆盖率加权。
7. 原始指标是所有有效后续帧的加权平均。
8. 当前默认归一化就是 `[0, 1]` 上的直接有界缩放。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 采样后的 prediction 视频帧 |
| 参考输入 | 采样后的 reference 视频帧 |
| 标注输入 | 来自 `mask_path` 或 `dyn_masks.npz` 的可选 ignore mask |
| 原始输出 | 一个 `[0, 1]` 范围内的加权平均标量 |
| 归一化输出 | 默认配置下与原始值相同的有界标量 |
| 细节字段 | `frame_scores`，其中包含 `frame_index`、`coverage`、`psnr_component`、`dino_component`、`score`，以及帧索引元数据 |
| 失败形式 | 如果无法形成有效的 novel-region 分数，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 集成说明

- 当前聚合状态：在 `image_static`、`image_dynamic`、`video_static` 中，只要成功产出数值就属于 `official`；纯 image reference 通常会因为没有多帧 reference 而为空。
- 它和 [`seen_region_preservation`](seen_region_preservation.md) 是互补的：前者看持续可见区域，后者看新暴露区域。
- 和 seen-region 指标一样，过小的 mask 会被跳过，而不是记为 0。
