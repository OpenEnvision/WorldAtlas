# Keystroke Partial Action Accuracy

**keystroke_partial_action_accuracy** 衡量每帧恢复动作与目标原子动作集合是否存在**非空交集**，用于容忍模型仅激活部分规定按键的情形。

记 \(P(\cdot)\) 将动作分解为原子分量（前进、后退、偏航、俯仰等）。在 \(T\) 帧对齐序列上：

\[
\mathrm{Acc}_{\mathrm{partial}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[P(\hat{g}_t) \cap P(g_t) \neq \emptyset]
\tag{1}
\]

无操作/无操作（no-op/no-op）情形计为有效部分匹配。越高越好。需要动作侧车与可恢复相机位姿。后端：`pose_action`。
