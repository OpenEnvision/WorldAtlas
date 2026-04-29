# 物体持久性

`object_permanence` 衡量的是：被 mask 标出来的动态前景物体，是否沿着参考 rollout 所暗示的外观轨迹保持为“同一个东西”。

## 这个指标测什么

- 前景物体是否在时间上保持可辨认的同一性
- prediction 的物体外观曲线是否跟 reference 的物体外观曲线一致
- 不是纯运动幅度指标，也不是整帧相似度

## 具体算法

1. 该指标要求存在可读取的参考视频，以及来自 `sample.mask_path` 或 `annotation_path/dyn_masks.npz` 的对齐动态 mask。
2. WorldArena 会先把 mask 序列对齐到采样后的参考帧。
3. 它会尽量选取采样窗口之前一帧作为 anchor object crop；如果拿不到，就退回到采样窗口的第一帧。
4. 对每个采样时刻：
    - 从 reference frame 中提取 masked object crop
    - 把 prediction frame resize 到 reference 分辨率后提取 masked object crop
    - 用 DINOv2 提取 anchor crop、reference crop、prediction crop 的特征
5. 然后计算两条相似度曲线：
    - `gt_similarity`：anchor crop vs reference crop
    - `prediction_similarity`：anchor crop vs prediction crop
6. 每帧 permanence 分数定义为 `1 - abs(prediction_similarity - gt_similarity)`，并裁剪到 `[0, 1]`。
7. 帧分数按 mask 覆盖率加权。
8. 原始指标是所有有效帧的加权平均。
9. 当前默认归一化就是 `[0, 1]` 上的直接有界缩放。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 采样后的 prediction 视频帧 |
| 参考输入 | 采样后的 reference 视频帧，以及选中的 anchor frame |
| 标注输入 | 来自 `mask_path` 或 `dyn_masks.npz` 的对齐动态 mask |
| 原始输出 | 一个 `[0, 1]` 范围内的 permanence 标量 |
| 归一化输出 | 默认配置下与原始值相同的有界标量 |
| 细节字段 | `anchor_frame_index`、`frame_scores`、`gt_similarity_curve`、`prediction_similarity_curve`，以及帧索引元数据 |
| 失败形式 | 如果 mask 不可用，或者无法形成有效 masked crop，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 集成说明

- 当前聚合状态：在 `image_static`、`image_dynamic`、`video_dynamic` 中，当 reference video 和对齐 mask 存在且成功产出数值时属于 `official`。
- 这个指标更偏向外观持久性，因此最好和 [`motion_accuracy`](motion_accuracy.md) 以及 [`pmf`](pmf.md) 一起看，分别覆盖“它还是不是同一个物体”和“它动得对不对”。
- mask 质量会直接影响分数。即使生成结果肉眼可接受，mask 对不齐也会拉低该指标。
