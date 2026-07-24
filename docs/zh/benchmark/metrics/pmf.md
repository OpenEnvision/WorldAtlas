# Physical Motion Fidelity

**pmf** 用来衡量预测视频是否保留了与参考 rollout 相近的整体运动动态，同时刻意降低对绝对空间位置和整体亮度变化的敏感性。
WorldArena 当前通过仓内的 NumPy FFT 实现直接计算 PMF。
该指标来自 PhysInOne [[Zhou et al., 2026](#reference)] 的 Fourier-based motion fidelity 思路：与其逐像素比较预测视频和参考视频，不如比较它们在空间-时间频率域中的归一化运动能量分布。

## 这个指标测什么

- 预测与参考之间的时空运动能量模式是否相近。
- 对简单空间/时间平移偏移和整体亮度缩放的稳健性。
- 生成 rollout 是否保留了参考视频的整体动态节奏和频率结构。

## 原理说明

引入 Fourier-based 指标的核心动机是：评估生成视频时，我们往往更关心运动动态是否相似，而不是每个像素值是否绝对对齐。
如果一段视频只是整体平移了起始位置、延后了运动开始时间，或者整体亮度被缩放，它的物理运动模式可能仍然和参考视频一致。
逐像素指标会强烈惩罚这些差异，但 PMF 希望尽量忽略这类初始状态和亮度差异，转而比较视频的时空频谱能量结构。

具体来说，PMF 会把视频看成一个 \(H\times W\times T\) 的时空信号，并对每个 RGB 通道做 3D DFT。
傅里叶系数的幅度描述某个空间频率和时间频率上有多少能量，phase 则编码位置、时间起点等平移信息。
在归一化能量谱中，整体亮度缩放会被分母抵消，简单时空平移只会改变 phase，而不会改变幅度能量分布。
因此，比较归一化能量谱可以更集中地评估 motion dynamics，而不是起始位置、绝对亮度或逐像素对齐。

## 具体算法

1. WorldArena 读取预测视频和参考视频的 rollout 评测窗口。
2. 如果两边采样到的帧数不同，就把它们都均匀重采样到较短的那一个长度。
3. 如果分辨率不同，就把预测帧 resize 到参考视频的分辨率。
4. 它把每个视频堆叠成形状为 `[T, H, W, C]` 的 RGB 张量。
5. 对每个 RGB 通道分别在 `H x W x T` 上做 3D 离散傅里叶变换。
6. 把傅里叶系数转成合并后的功率谱：
    - `power(u, v, s) = sum_c |F_c(u, v, s)|^2`
7. 再把功率谱归一化成类似概率分布的能量分布：
    - `E(u, v, s) = power(u, v, s) / sum power`
8. 计算预测与参考能量分布之间的 total variation 距离：
    - `tv = 0.5 * sum |E_pred - E_ref|`
9. 最终输出：
    - `pmf = -log(max(tv, 1e-12))`
10. WorldArena 再做有界归一化：
    - `lower = 0.0`
    - `upper = 6.0`
    - 越高越好

PMF 越高，表示两段视频在运动能量结构上越相似。若频谱完全一致，当前实现会被 epsilon floor 截断，而不是返回数学意义上的无穷大。

## 频谱定义

给定视频 \(V\)，其 RGB 通道记为 \(c\in\{0,1,2\}\)，空间和时间坐标为 \((h,w,t)\)。
对每个通道做 3D 离散傅里叶变换：

!!! info "3D DFT"

    \[
    \tilde{V}(c;u,v,s)
    =
    \sum_{h=0}^{H-1}
    \sum_{w=0}^{W-1}
    \sum_{t=0}^{T-1}
    V(c;h,w,t)
    \exp
    \left(
        -2\pi i
        \left(
            \frac{uh}{H}
            +
            \frac{vw}{W}
            +
            \frac{st}{T}
        \right)
    \right)
    \]

其中，\((u,v)\) 是空间频率索引，\(s\) 是时间频率索引。
幅度和相位分别为：

!!! info "幅度与相位"

    \[
    A_{c;u,v,s}
    =
    \left|
        \tilde{V}(c;u,v,s)
    \right|,
    \qquad
    \psi_{c;u,v,s}
    =
    \arg
    \left(
        \tilde{V}(c;u,v,s)
    \right)
    \]

PMF 使用的是跨 RGB 通道合并后的归一化能量谱：

!!! info "归一化能量谱"

    \[
    E_{u,v,s}
    =
    \frac{
        \sum_{c=0}^{2}
        \left|
            \tilde{V}(c;u,v,s)
        \right|^2
    }{
        \sum_{u',v',s'}
        \sum_{c=0}^{2}
        \left|
            \tilde{V}(c;u',v',s')
        \right|^2
    }
    \]

这让 \(E\) 像一个分布一样描述视频的时空运动能量集中在哪些频率上。

## 不变性直觉

考虑一个经过时空平移和亮度缩放的视频：

!!! info "平移与亮度缩放"

    \[
    V'(c;h,w,t)
    =
    \lambda
    V(c;h+\delta_h,w+\delta_w,t+\delta_t),
    \qquad
    \lambda\in(0,1]
    \]

在 duplicate padding 等边界假设下，傅里叶变换满足：

!!! info "频域变化"

    \[
    \tilde{V}'(c;u,v,s)
    =
    \lambda
    e^{i\delta\psi}
    \tilde{V}(c;u,v,s)
    \]

其中：

!!! info "相位偏移"

    \[
    \delta\psi
    =
    2\pi
    \left(
        \frac{u\delta_h}{H}
        +
        \frac{v\delta_w}{W}
        +
        \frac{s\delta_t}{T}
    \right)
    \]

因此，时空平移只改变 phase，整体亮度缩放只给幅度乘上统一因子。
当我们使用归一化能量谱 \(E_{u,v,s}\) 时，亮度缩放的 \(\lambda^2\) 会在分子和分母中抵消；平移导致的相位项也不会影响幅度平方。
这就是 PMF 对简单时空平移和亮度缩放更稳健的原因。

## PMF 公式

给定生成视频 \(V_{\mathrm{gen}}\) 和参考视频 \(V_{\mathrm{ref}}\)，先分别计算归一化能量谱 \(E_{\mathrm{gen}}\) 和 \(E_{\mathrm{ref}}\)。
两者之间的 total variation 距离为：

!!! info "Total Variation"

    \[
    d_{\mathrm{TV}}
    \left(
        E_{\mathrm{gen}},
        E_{\mathrm{ref}}
    \right)
    =
    \frac{1}{2}
    \sum_{u,v,s}
    \left|
        E^{\mathrm{gen}}_{u,v,s}
        -
        E^{\mathrm{ref}}_{u,v,s}
    \right|
    \]

PMF 定义为：

!!! info "Physical Motion Fidelity"

    \[
    \operatorname{PMF}
    \left(
        V_{\mathrm{gen}},
        V_{\mathrm{ref}}
    \right)
    =
    -\ln
    \left(
        \max
        \left(
            d_{\mathrm{TV}}
            \left(
                E_{\mathrm{gen}},
                E_{\mathrm{ref}}
            \right),
            10^{-12}
        \right)
    \right)
    \]

TV 距离越小，说明两段视频的归一化时空频谱越相似，PMF 越高。
当前实现使用 `epsilon = 1e-12` 防止完全一致时出现 \(-\ln(0)\)。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | rollout 窗口内的预测视频 |
| 参考输入 | rollout 窗口内的参考视频 |
| 标注输入 | 无 |
| 原始输出 | 单个 PMF 标量 |
| 归一化输出 | 经过有界缩放后的单个 `[0, 1]` 标量 |
| 细节字段 | `tv_distance`、`epsilon_floor`、`aligned_frame_count`、`reference_resolution`、`prediction_frame_count`、`reference_frame_count`、`temporal_alignment`、`prediction_resized`、`reference_resized`，以及帧索引元数据 |
| 失败形式 | 如果任一视频少于两帧，或者解码/对齐失败，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 在 `image_static`、`image_dynamic`、`video_dynamic` 中，只要产出数值分数，就是 `official` |
| 论文可用性结论 | `主表可用，但需要说明它是 reference-based Fourier dynamics 指标` |

- 当你关心生成 rollout 是否跟参考视频保持相近的整体物理动态时，它明显比单纯的 motion amplitude 指标更有信息量。
- 对纯 image reference，当前实现通常会因为 reference 少于 2 帧而返回空值；只有存在兼容多帧 reference 时才会出分。
- 但它仍然不是完整的 correctness 指标。两段视频即便拥有相近的运动频谱，也可能在对象身份、因果交互或语义正确性上完全不同。
- 因为它依赖 paired reference，所以更适合和 control、persistence、quality 指标一起报告，而不是单独拿来宣称 world modeling 已经做对。

## 参考文献 {#reference}

!!! quote "PhysInOne: Visual Physics Learning and Reasoning in One Suite"

    Siyuan Zhou, Hejun Wang, Hu Cheng, Jinxi Li, Dongsheng Wang, Junwei Jiang, Yixiao Jin, Jiayue Huang, Shiwei Mao, Shangjia Liu, Yafei Yang, Hongkang Song, Shenxing Wei, Zihui Zhang, Peng Huang, Shijie Liu, Zhengli Hao, Hao Li, Yitian Li, Wenqi Zhou, Zhihan Zhao, Zongqi He, Hongtao Wen, Shouwang Huang, Peng Yun, Bowen Cheng, Pok Kazaf Fu, Wai Kit Lai, Jiahao Chen, Kaiyuan Wang, Zhixuan Sun, Ziqi Li, Haochen Hu, Di Zhang, Chun Ho Yuen, Bing Wang, Zhihua Wang, Chuhang Zou, and Bo Yang.  
    arXiv:2604.09415, 2026.  
    [https://arxiv.org/abs/2604.09415](https://arxiv.org/abs/2604.09415)

??? note "BibTeX"

    ```bibtex
    @misc{zhou2026physinonevisualphysicslearning,
        title = {PhysInOne: Visual Physics Learning and Reasoning in One Suite},
        author = {Siyuan Zhou and Hejun Wang and Hu Cheng and Jinxi Li and Dongsheng Wang and Junwei Jiang and Yixiao Jin and Jiayue Huang and Shiwei Mao and Shangjia Liu and Yafei Yang and Hongkang Song and Shenxing Wei and Zihui Zhang and Peng Huang and Shijie Liu and Zhengli Hao and Hao Li and Yitian Li and Wenqi Zhou and Zhihan Zhao and Zongqi He and Hongtao Wen and Shouwang Huang and Peng Yun and Bowen Cheng and Pok Kazaf Fu and Wai Kit Lai and Jiahao Chen and Kaiyuan Wang and Zhixuan Sun and Ziqi Li and Haochen Hu and Di Zhang and Chun Ho Yuen and Bing Wang and Zhihua Wang and Chuhang Zou and Bo Yang},
        year = {2026},
        eprint = {2604.09415},
        archivePrefix = {arXiv},
        primaryClass = {cs.CV},
        url = {https://arxiv.org/abs/2604.09415},
    }
    ```
