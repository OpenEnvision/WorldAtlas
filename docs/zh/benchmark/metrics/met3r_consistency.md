# MEt3R 一致性

`met3r_consistency` 是一个基于被包装 MEt3R 运行时的诊断性多视角一致性探针。

<figure markdown="span">
  ![MEt3R 方法概览](../../../assets/img/met3r_method_overview.jpg)
  <figcaption>MEt3R 使用多视角生成图像之间的几何和表征一致性来度量生成结果的 multi-view consistency。</figcaption>
</figure>

## 这个指标测什么 {#what-it-measures}

- 在 MEt3R 打分模型下，参考锚点视角和预测锚点视角有多接近
- 参考与预测 anchor view 之间的 multi-view / view-to-view 一致性
- 不是 headline metric，也不是几何真值监督的替代品

## 具体算法 {#algorithm}

1. WorldArena 会在 benchmark protocol 的 anchor 位置上，把 reference anchor frames 与 prediction anchor frames 配对。
2. MEt3R 源码树会从当前配置好的 MEt3R checkout 中解析出来。
3. 运行时会读取当前配置的模型选项；默认包括：
    - `backbone = mast3r`
    - `distance = cosine`
    - `batch_size = 1`
4. 成对的 anchor frame 会分 batch 送入被包装的 MEt3R 模型。
5. runtime 会为每一对 anchor view 返回一个分数。
6. 原始指标就是这些 pair score 的平均值。
7. 归一化会遵循当前 distance 的方向：
    - 默认 `cosine` 距离使用 `[0, 2]` 的 benchmark 范围
    - 默认配置下是“越小越好”

## 输入与输出 {#inputs-outputs}

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | prediction anchor frames |
| 参考输入 | reference anchor frames |
| 标注输入 | 无 |
| 原始输出 | 一个 MEt3R 均值标量 |
| 归一化输出 | 经过 distance-aware 缩放后的 `[0, 1]` 标量 |
| 细节字段 | `frame_scores`、`pair_count`、`source_root`、`device`、`batch_size`、`img_size`、`use_norm`、`backbone`、`feature_backbone`、`feature_backbone_weights`、`upsampler`、`distance`、`freeze`、`native_direction` |
| 失败形式 | 如果 checkout、submodule 或旁路运行时没有准备好，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 集成说明 {#integration}

- 当前聚合状态：在所有 formal suite 中都属于 `diagnostic`。
- wrapper 会按 prediction 路径和 runtime 选项缓存结果，因此同一视频重复访问时，首轮之后成本较低。

## 参考文献 {#reference}

!!! quote "MEt3R: Measuring Multi-View Consistency in Generated Images"

    Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen.  
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

??? note "BibTeX"

    ```bibtex
    @inproceedings{asim25met3r,
        title = {MEt3R: Measuring Multi-View Consistency in Generated Images},
        author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
        booktitle = {IEEE/CVF Computer Vision and Pattern Recognition ({CVPR})},
        year = {2025},
    }
    ```
