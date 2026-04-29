# Optical Flow AEPE

`optical_flow_aepe` 用来衡量生成视频的 photometric consistency，也就是模型能否在连续帧之间生成稳定、一致的视觉外观。
它通过估计相邻帧之间的前向和后向光流，计算点在“前向追踪再反向追踪”之后能否回到原始位置。
误差越小，说明画面中的纹理、身份、局部结构和可见区域在短时间内越稳定；误差越大，通常意味着存在纹理闪烁、身份漂移、物体消失或局部外观突变等不一致现象。

## 这个指标测什么

- 相邻帧之间的短程视觉稳定性。
- 前向光流和后向光流是否形成一致的 cycle。
- 生成对象的身份、纹理和局部外观是否在连续帧中保持稳定。

## 原理说明

给定连续两帧 \(A\) 和 \(B\)，指标先在中心裁剪区域内采样一组点 \(p_A\)。
中心裁剪可以减少边界区域由于遮挡、画幅外运动或光流外推带来的噪声，使分数更集中地反映主体区域的外观一致性。

随后，指标使用从 \(A\) 到 \(B\) 的前向光流 \(F_{A\rightarrow B}\) 把这些点追踪到下一帧：

!!! info "前向追踪"

    \[
    p_B
    =
    p_A
    +
    F_{A\rightarrow B}(p_A)
    \]

接着，再使用从 \(B\) 到 \(A\) 的后向光流 \(F_{B\rightarrow A}\) 把同一批点追踪回原始帧：

!!! info "反向追踪"

    \[
    p'_A
    =
    p_B
    +
    F_{B\rightarrow A}(p_B)
    \]

如果生成视频在视觉外观上是稳定的，同一个局部纹理或物体点从 \(A\) 追踪到 \(B\)，再从 \(B\) 追踪回 \(A\) 后，应当接近原始位置，即 \(p'_A \approx p_A\)。
如果视频中发生了纹理闪烁、物体身份变化、局部结构漂移或物体突然消失，正反向追踪就很难闭合，最终的 endpoint error 会变大。

## 具体算法

1. WorldArena 会把整段预测视频解码成 PNG 帧。
2. 对每一对相邻帧 `(I_t, I_{t+1})`，被包装的第三方指标会运行两次 `SEA-RAFT` [[Wang et al., 2024](#reference)]：
    - 前向光流 `f_{t->t+1}`
    - 后向光流 `f_{t+1->t}`
3. 两个光流场都会被裁成中心区域，参数是 `crop = 30`。
4. 对裁剪区域里的每个采样点 \(p_{A,i}\)，指标执行一次 round trip：
    - 先用前向光流 \(F_{A\rightarrow B}(p_{A,i})\) 从第 \(A\) 帧追踪到第 \(B\) 帧。
    - 再在目标位置采样后向光流 \(F_{B\rightarrow A}(p_{B,i})\)，把点追踪回第 \(A\) 帧。
    - 比较返回位置 \(p'_{A,i}\) 和原始位置 \(p_{A,i}\) 的欧氏距离。
5. 单对帧的 photometric consistency error 定义为全部采样点的 Average End-Point Error（AEPE）。
6. 原始指标值是所有相邻帧对的 AEPE 平均值。
7. WorldArena 用 `empirical_min = 0.0`、`empirical_max = 1.192`、`higher_is_better = false` 做归一化。

## 误差公式

对一对相邻帧，令中心裁剪区域内共有 \(N\) 个采样点。
第 \(i\) 个点的原始位置为 \(p_{A,i}\)，经过前向和后向光流追踪后返回的位置为 \(p'_{A,i}\)。
该帧对的 photometric consistency error 为：

!!! info "核心公式"

    \[
    e_{\mathrm{photometric}}
    =
    \frac{1}{N}
    \sum_{i=1}^{N}
    \left\lVert
        p_{A,i}
        -
        p'_{A,i}
    \right\rVert_2
    \]

其中，\(N\) 是中心裁剪区域内参与计算的采样点数量；\(\lVert\cdot\rVert_2\) 表示二维像素位置之间的欧氏距离。
最终视频级误差会对整段生成视频中的所有相邻帧对继续求平均；模型级 photometric consistency error 则在评测集内所有生成视频上继续聚合。
AEPE 越高，表示正反向光流越难闭合，视觉外观越不稳定。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 无 |
| 标注输入 | 无 |
| 原始输出 | 单个前后向一致性误差标量 |
| 归一化输出 | 单个 `[0, 1]` 标量，越高越好 |
| 细节字段 | `frame_count`、`native_score` |
| 失败形式 | 如果 SEA-RAFT runtime 不可用，则返回 `raw = null`、`normalized = null` 和错误字符串 |

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
