# Motion Magnitude

`motion_magnitude` 用来衡量预测结果里“有多少运动”，而不关心这些运动是否正确。
它主要用于识别过于保守的生成模型：有些模型为了保持画面稳定和高质量，会只生成非常轻微的运动，甚至在 prompt 明确要求运动时仍输出近似静态的视频。
`motion_magnitude` 通过相邻帧之间的光流幅值来检测这种 motion under-generation。

## 这个指标测什么

- 预测结果整体的运动幅度。
- 相邻帧对上的光流强度。
- 生成视频是否接近静态，或者是否确实产生了可观测运动。
- 它不是 motion correctness，也不是 smoothness；运动很大不代表运动方向、对象或语义正确。

## 原理说明

视频生成模型常见的一种失败模式是“保守生成”：画面看起来清晰、稳定、没有明显闪烁，但物体几乎不动，或者只有非常微弱的局部变化。
这类结果在静态图像质量指标上可能表现不错，在 motion smoothness 上也可能得分较高，因为静止视频天然容易被插值。
但如果任务本身要求动态变化，这种近静态输出就不符合预期。

`motion_magnitude` 因此直接测量生成视频内部的运动量。
它对每一对相邻帧估计稠密光流，并将每个像素的二维光流向量转换成幅值。
单对帧的运动强度取整张光流幅值图的中位数，而不是最大值或均值，这样可以减少少量离群光流、边界噪声或局部伪影的影响。
最后对整段视频所有相邻帧对求平均，得到视频级运动幅度。

## 具体算法

1. WorldArena 会把整段预测视频解码成 PNG 帧。
2. 对每一对相邻帧，被包装的第三方指标会运行 `SEA-RAFT` [[Wang et al., 2024](#reference)] 估计光流。
3. 它把光流转成幅值图 `m = sqrt(u^2 + v^2)`。
4. 单对帧统计量定义为这张幅值图在所有像素上的中位数。
5. 原始指标值是所有相邻帧对统计量的平均值。
6. WorldArena 使用当前 z-score 校准：
    - `avg = 3.2425`
    - `std = 3.4505`
    - `z_min = -0.8638`
    - `z_max = 2.7498`
    - 输出范围 `[0.25, 0.75]`

## 误差公式

给定相邻两帧 \((I_t, I_{t+1})\)，SEA-RAFT 估计出的光流为：

!!! info "光流场"

    \[
    \mathbf{F}_t(x,y)
    =
    (u_t(x,y), v_t(x,y))
    \]

对应的光流幅值图为：

!!! info "光流幅值"

    \[
    F_t(x,y)
    =
    \sqrt{
        u_t(x,y)^2
        +
        v_t(x,y)^2
    }
    \]

单对帧的 motion magnitude 定义为所有像素光流幅值的中位数：

!!! info "单对帧运动幅度"

    \[
    s^{(t)}_{\mathrm{motion\text{-}mag}}
    =
    \operatorname{median}_{(x,y)}
    F_t(x,y)
    \]

最终原始指标值是所有相邻帧对的平均值：

!!! info "最终原始分数"

    \[
    s_{\mathrm{motion\text{-}mag}}
    =
    \frac{1}{T-1}
    \sum_{t=0}^{T-2}
    s^{(t)}_{\mathrm{motion\text{-}mag}}
    \]

分数越高，表示生成视频中整体运动越明显。
如果模型输出近静态视频，光流幅值中位数会接近零，最终分数也会较低。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 在 official backend 下不使用参考输入 |
| 标注输入 | 无 |
| 原始输出 | 单个平均运动幅度标量 |
| 归一化输出 | 经过 z-score 映射后的单个 `[0, 1]` 标量 |
| 细节字段 | `frame_count`、`native_score`、`frame_indices`、`sampled_frame_indices` |
| 失败形式 | 如果光流 backend 不可用，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 只要产出数值分数，就是 `official` |
| 论文可用性结论 | `主表可用，但需要说明它只衡量运动量` |

- 这个指标很适合区分“基本不动”的结果和“确实动起来了”的结果。
- 但它并不回答“动得对不对”。一个运动幅度很大但完全错误的视频也可以得高分。
- 它也不衡量运动是否平滑；运动量大但闪烁严重的视频仍可能得高分。
- 建议用法：作为 dynamics / motion amount 的主要指标之一，并和 `motion_accuracy`、`motion_smoothness` 一起解读。

## 参考文献 {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Yihan Wang, Lahav Lipson, and Jia Deng.  
    European Conference on Computer Vision (ECCV), 2024.

??? note "BibTeX"

    ```bibtex
    @inproceedings{wang2024sea,
        title = {Sea-raft: Simple, efficient, accurate raft for optical flow},
        author = {Wang, Yihan and Lipson, Lahav and Deng, Jia},
        booktitle = {European Conference on Computer Vision},
        pages = {36--54},
        year = {2024},
        organization = {Springer},
    }
    ```
