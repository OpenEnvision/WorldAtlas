# Image Quality（MUSIQ）

`image_quality` 是正式的无参考 `image_static` quality 指标。它使用 MUSIQ-SPAQ 图像质量模型评估 prediction 帧，不需要 paired reference rollout。

对协议选出的每个 prediction 帧 $I_i$，MUSIQ 输出原生 $[0,100]$ 分数 $q_i$。WorldArena 先逐帧归一化并截断：

\[
s_i = \mathrm{clip}\!\left(\frac{q_i}{100}, 0, 1\right),
\]

再对归一化帧分数取平均：

\[
S_{\mathrm{MUSIQ}} = \frac{1}{N}\sum_{i=1}^{N}s_i.
\]

得分越高越好。该结果进入 `per_metric_summary` 与七项等权 quality 总分。默认 backend 为 `musiq`，预处理模式为 `longer`。
