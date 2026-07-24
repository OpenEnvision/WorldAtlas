# 片段连续性（Segment Continuity）

片段连续性度量生成视频是否出现**检测到的场景切换**。非预期的硬切会破坏时序连贯性，反映片段不连续。

TransNetV2 为解码帧 $x_i$ 输出场景边界概率 $p_i$。当 $p_i > \theta$ 时标记为切点候选（默认 $\theta = 0.5$）。切点记录遵循 TransNetV2 的 0→1 跃迁解析：在帧索引 $i$ 处由 $0$ 升至 $1$ 即记为一次切换。

为抑制误检，切点候选需满足最小场景长度 $L$（默认 $L = 10$ 帧）：任一保留切点两侧的片段长度均不得短于 $L$。设单视频保留切点数为 $n_{\mathrm{cut}}$，则 per-video 得分为二值：

\[
S_{\mathrm{seg}}
=
\begin{cases}
1, & n_{\mathrm{cut}} = 0 \\
0, & n_{\mathrm{cut}} > 0
\end{cases}
\tag{1}
\]

对 $N$ 个视频，模型级聚合为

\[
\bar{S}_{\mathrm{seg}}
=
1 - \frac{n_{\mathrm{cut}}}{N}
\tag{2}
\]

其中 $n_{\mathrm{cut}}$ 统计至少含一个保留切点的视频数。WorldArena 将式 (1) 的二值结果存为 per-sample raw 分数。该指标**仅需 prediction**，且依赖 TransNetV2 推理代码与权重。

## 参考文献 {#reference}

!!! quote "TransNet V2 — Shot Boundary Detection"

    Sočik et al., ACM MM 2020.
    [https://arxiv.org/abs/2008.04838](https://arxiv.org/abs/2008.04838)
