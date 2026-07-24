# 场景记忆 F1

**scene_memory_f1** 衡量相机回访之后，先前观测到的场景几何是否仍被几何上支持。与图像对相似度指标不同，它通过 observation 点云与 revisit 点云之间的 **3D 重叠** 来评分场景记忆。WorldArena 在包含“离开后再返回”动作的长序列套件上，通过 `point_cloud_f1` 后端实现该指标。

长时域世界模型需要在 extended rollout 中保持场景结构的稳定。模型可能在相邻帧之间看起来仍然一致，但在视点回到先前区域时，墙面、家具或空间布局已经发生变化。Scene Memory F1 直接针对这类失败：它考察回访后有多少先前观测点仍能在 3D 中被匹配，以及 revisit 点云中有多少几何缺乏先前观测的支持。

Runtime 首先解析 observation 与 revisit 点云。若存在显式的 observation/revisit sidecar，或 bundled 的 `scene_memory` NPZ 归档，则直接加载对应点云；否则根据 action payload 及其关联的 ViPE depth sidecar 重建两侧点云。无效点会被剔除，并在配置 `voxel_size` 时进行体素下采样。当 `centroid_align = true`（默认值）时，revisit 点云会先平移，使其质心与 observation 点云质心对齐，再进入打分。距离阈值 \(\tau_d\) 优先读取 `distance_threshold`；若未配置，则使用 `distance_threshold_ratio * scene_diagonal`，默认比例为 `0.02`。

令 \(P_{obs}\) 为 observation 点云，\(P_{rev}\) 为对齐后的 revisit 点云。对点 \(p\) 与点云 \(P\)，记 \(d(p, P)\) 为 \(p\) 到 \(P\) 中最近邻点的距离。**Retention（保留率）** 衡量先前观测在回访时仍被支持的比例：

\[
R_{mem}
=
\frac{
    |\{p \in P_{obs}: d(p, P_{rev}) < \tau_d\}|
}{
    |P_{obs}|
}
\tag{1}
\]

**Hallucination（幻觉率）** 衡量 revisit 中缺乏 observation 支持的几何。它在 F1 聚合前先转换为类似 precision 的项：

\[
R_{hall}
=
\frac{
    |\{q \in P_{rev}: d(q, P_{obs}) \ge \tau_d\}|
}{
    |P_{rev}|
}
\tag{2}
\]

\[
P_{mem} = 1 - R_{hall}
\tag{3}
\]

最终的场景记忆分数对 retention 与 precision 做对称组合：

\[
S_{scene}
=
\frac{
    2P_{mem}R_{mem}
}{
    P_{mem}+R_{mem}
}
\tag{4}
\]

\(S_{scene}\) 越高，表示场景记忆越强：更多先前观测几何被保留，且 revisit 中 unsupported 的点更少。该指标评估的是跨 revisit 的几何持续性，而非光度相似度、语义 prompt 对齐或绝对 metric-scale 重建精度。
