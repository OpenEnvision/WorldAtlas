# Camera Error

`camera_error` 用来衡量从预测视频恢复出的相机轨迹，与标注相机位姿之间有多接近。

## 这个指标测什么

- 相对位姿标注的相机运动保真度
- 视角控制，而不是物体运动控制
- 在后对齐之后与标注 extrinsics 的一致程度

## 具体算法

1. WorldArena 会把预测视频解码成 PNG 帧，并统计全部可解码帧数。
2. `load_camera_matrices()` 从样本标注目录加载 `poses.npy`。如果存储的位姿数量和解码帧数不一致，则通过以下方式重采样：
    - 平移做线性插值
    - 旋转做 quaternion slerp
    - 如果存在 `indexes.txt`，优先使用其中的位置索引
3. 被包装的第三方指标会在这些解码帧上运行 `DROID-SLAM` [[Teed and Deng, 2021](#reference)]，从预测结果中恢复相机轨迹。
4. 第三方实现随后会把恢复出的轨迹转换到它评估代码期望的坐标系。
5. 平移项会通过最小二乘尺度 `s` 做 scale-invariant 对齐。
6. 旋转误差是对齐后的平均相对旋转角度，单位是度。
7. 平移误差是尺度对齐后的平均欧氏距离。
8. WorldArena 会对两个分量分别归一化：
    - 旋转误差范围 `[0.0, 15.0]`，越小越好
    - 平移误差范围 `[0.0, 0.5]`，越小越好
9. 最终归一化分数是两个归一化分量的几何平均。

## 误差公式

对第 \(i\) 帧，标注位姿记为 \((R_i^{gt}, t_i^{gt})\)，DROID-SLAM [[Teed and Deng, 2021](#reference)] 从生成视频恢复出的位姿记为 \((R_i, t_i)\)。其中 \(R_i^{gt}, R_i \in SO(3)\)，\(t_i^{gt}, t_i \in \mathbb{R}^3\)。

!!! info "核心公式"

    旋转项使用两个相机旋转矩阵之间的角度偏差：

    \[
    e_{\theta,i}
    =
    \arccos\!\left(
        \frac{\operatorname{tr}\!\left(R_i^{gt} R_i^\top\right)-1}{2}
    \right)
    \cdot \frac{180}{\pi}
    \]

    平移项先用全部有效帧估计一个最小二乘尺度，再计算尺度不变欧氏距离：

    \[
    s
    =
    \arg\min_{\alpha}
    \sum_{i=1}^{N}
    \left\| t_i^{gt} - \alpha t_i \right\|_2^2,
    \qquad
    e_{t,i}
    =
    \left\| t_i^{gt} - s t_i \right\|_2
    \]

单个视频的原始 camera error 分量是全部有效帧的均值：

\[
e_{\theta}
=
\frac{1}{N}
\sum_{i=1}^{N} e_{\theta,i},
\qquad
e_t
=
\frac{1}{N}
\sum_{i=1}^{N} e_{t,i}
\]

统计模型级相机可控性误差时，再对完整评测集内所有生成视频、所有有效帧继续求平均。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 无 |
| 标注输入 | `poses.npy`；如果要重采样位姿，可选 `indexes.txt` |
| 原始输出 | `{"rotation_error": float, "translation_error": float}` |
| 归一化输出 | 经过分量归一化和几何平均之后的单个 `[0, 1]` 标量 |
| 细节字段 | `frame_count`、`camera_scale`、`native_components` |
| 失败形式 | 如果位姿数据缺失或 DROID runtime 失败，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成说明

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 只要存在位姿标注且成功产出数值分数，就是 `official` |
| 指标定位 | `core camera-control metric` |
| 论文可用性结论 | 可作为核心相机可控性指标使用，但需要说明固定内参和尺度不变对齐假设 |

- 当前被包装的 DROID-SLAM [[Teed and Deng, 2021](#reference)] 路径遵循 上游设定，在第三方实现内部使用**固定的 synthetic intrinsics**（`[500, 500, 256, 256]`，再按帧大小缩放）。
- 这个设定不把指标解释为逐样本真实标定条件下的绝对相机重建误差，而是解释为固定内参假设下、内部自洽的 camera controllability error。
- 平移分量使用最小二乘尺度做后对齐，因此该指标更强调相机运动方向、旋转控制和尺度不变的轨迹一致性，而不是严格的 metric-space 尺度保真。
- 建议报告方式：作为核心 camera-control metric 使用，并在方法说明中明确它是基于 DROID-SLAM、固定 synthetic intrinsics 和 scale-invariant alignment 的相机可控性误差。

## 参考文献 {#reference}

!!! quote "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Zachary Teed and Jia Deng.  
    Advances in Neural Information Processing Systems (NeurIPS), 2021.

??? note "BibTeX"

    ```bibtex
    @article{teed2021droid,
        title = {Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras},
        author = {Teed, Zachary and Deng, Jia},
        journal = {Advances in neural information processing systems},
        volume = {34},
        pages = {16558--16569},
        year = {2021},
    }
    ```
