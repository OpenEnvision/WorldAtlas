# Motion Accuracy

`motion_accuracy` 用来衡量生成视频中的运动是否主要发生在标注的动态对象区域内。
它依赖第一帧动态对象 mask，并用 SAM2 [[Ravi et al., 2024](#reference-sam2)] 将这些 mask 传播到整段视频；随后用 SEA-RAFT [[Wang et al., 2024](#reference-searaft)] 估计相邻帧光流，比较 mask 内外的最大运动幅值。
直观地说，如果应该运动的前景对象确实比背景区域动得更强，那么该指标会更高。

## 这个指标测什么

- 被跟踪前景区域是否比背景动得更强。
- 受第一帧动态对象 mask 条件约束的运动显著性。
- 生成视频的运动是否集中在 intended dynamic regions。
- 它不是对目标轨迹、目标方向或目标速度曲线的严格比较，也不验证运动语义是否完全正确。

## 原理说明

很多视频生成任务并不只要求“画面里有运动”，还要求运动发生在正确的对象或区域上。
例如，如果 prompt 或数据设定要求前景车辆移动，而背景建筑保持稳定，那么理想情况下，动态对象 mask 内的光流幅值应该高于背景区域。

`motion_accuracy` 就是围绕这个想法设计的。
它先使用数据集中提供的第一帧动态对象 mask 作为监督信号，再用 SAM2 将这些对象 mask 传播到后续帧。
随后，对每一对相邻生成帧运行 SEA-RAFT 得到稠密光流，并把光流向量转换为幅值图。
最后，指标比较 mask 内最大运动和 mask 外最大运动：如果 mask 内最大光流更大，说明主要运动更可能发生在预期对象区域；如果背景区域最大光流更大，说明可能存在背景漂移、错误运动或目标没有按预期移动。

这个指标只使用 prediction video、第一帧动态对象 mask 和光流估计，不使用参考视频。
因此它更像是一个 mask-conditioned motion saliency 指标，而不是完整的 motion faithfulness 指标。

## 具体算法

1. WorldArena 要求样本提供 `mask_dir`，并把目录里的每一张 mask 图像都作为初始 object mask 读入。
2. 被包装的第三方指标会构建一个 `SAM2` [[Ravi et al., 2024](#reference-sam2)] 视频预测器，并且只把这些 mask 当作第一帧监督。
3. `SAM2` 会把这些 mask 在整段视频里传播。当前实现还会把每个传播后的 mask 再向外膨胀 `10` 个像素。
4. 对每一对相邻帧 `(I_t, I_{t+1})`，指标都会运行 `SEA-RAFT` [[Wang et al., 2024](#reference-searaft)] 估计光流。
5. 它把光流转成幅值图 `m_t = sqrt(u^2 + v^2)`。
6. 用该时刻传播得到的 mask，把幅值图分成前景和背景两部分，并计算：
    - `object_max = max(m_t inside mask)`
    - `background_max = max(m_t outside mask)`
7. 单对帧分数定义为 `pair_score = object_max - background_max`。
8. 原始指标值是所有相邻帧对分数的平均值。
9. WorldArena 对该分数使用当前 z-score 校准：
    - `avg = -0.1965`
    - `std = 2.3687`
    - `z_min = -1.6147`
    - `z_max = 1.5180`
    - 输出范围 `[0.25, 0.75]`

## 误差公式

给定相邻两帧 \((I_t, I_{t+1})\)，SEA-RAFT 估计出的光流为：

!!! info "光流场"

    \[
    \mathbf{F}_t(x,y)
    =
    (u_t(x,y), v_t(x,y))
    \]

将二维光流转成幅值图：

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

令 \(M_t\in\{0,1\}^{H\times W}\) 表示由 SAM2 传播得到的第 \(t\) 帧动态对象 mask，\(\bar{M}_t=1-M_t\) 表示背景区域。
单对帧的 motion accuracy score 定义为：

!!! info "单对帧分数"

    \[
    s^{(t)}_{\mathrm{motion\text{-}acc}}
    =
    \max(F_t\odot M_t)
    -
    \max(F_t\odot \bar{M}_t)
    \]

其中，\(\odot\) 表示逐像素乘法，\(\max(\cdot)\) 表示在矩阵所有元素中取最大值。
最终原始分数是所有相邻帧对的平均值：

!!! info "最终原始分数"

    \[
    s_{\mathrm{motion\text{-}acc}}
    =
    \frac{1}{T-1}
    \sum_{t=0}^{T-2}
    s^{(t)}_{\mathrm{motion\text{-}acc}}
    \]

分数越高，表示动态对象区域内的最大运动越明显地超过背景区域的最大运动。
如果背景发生强烈漂移、相机抖动或 mask 外区域出现异常运动，该分数会降低。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 无 |
| 标注输入 | 第一帧 mask 目录 |
| 原始输出 | 单个前景减背景的平均运动分数标量 |
| 归一化输出 | 经过 z-score 映射后的单个 `[0, 1]` 标量 |
| 细节字段 | `frame_count`、`mask_count`、`mask_dir`、`generate_type`、runtime device 细节、`native_score` |
| 失败形式 | 如果 mask 不存在，或者重型 runtime 失败，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 只要存在 mask 且成功产出数值分数，就是 `official` |
| 论文可用性结论 | `主表可用，但需要说明 mask-conditioned 定义` |

- 当前分数并不是严格的 motion-faithfulness 指标。它主要衡量 mask 内运动是否比背景更显著，不能验证目标运动方向、速度曲线或动作语义是否完全正确。
- 由于公式使用最大光流幅值，分数会受到离群光流、局部伪影或背景大幅运动影响。
- 当前 WorldArena 路径依赖数据集提供的第一帧 mask；如果没有可靠 mask，该指标不应被解释为有效 motion-control 结果。
- 这个指标依赖第一帧 mask 传播，因此一旦 `SAM2` 传播失败，分数就会被直接污染。
- 建议用法：作为 action-control / motion-control 维度的主要 mask-conditioned 指标之一，并和 `motion_magnitude`、`motion_smoothness`、`object_permanence` 一起解读。

## 参考文献 {#reference-searaft}

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

### SAM2 {#reference-sam2}

!!! quote "SAM 2: Segment Anything in Images and Videos"

    Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer.  
    arXiv:2408.00714, 2024.

??? note "BibTeX"

    ```bibtex
    @article{ravi2024sam2,
        title = {SAM 2: Segment Anything in Images and Videos},
        author = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
        journal = {arXiv preprint arXiv:2408.00714},
        year = {2024},
    }
    ```
