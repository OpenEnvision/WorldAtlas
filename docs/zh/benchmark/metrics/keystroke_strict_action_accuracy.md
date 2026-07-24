# Keystroke Strict Action Accuracy

**keystroke_strict_action_accuracy** 衡量恢复动作与目标离散键盘动作标签在**逐帧上的精确一致**。从预测恢复的相机位姿（ViPE 侧车）经阈值扫描离散化为原子动作集合，与 action payload 中的目标动作序列对比。

记 \(\hat{g}_t\)、\(g_t\) 分别为帧 \(t\) 的恢复与目标动作集合。在 \(T\) 帧对齐序列上：

\[
\mathrm{Acc}_{\mathrm{strict}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[\hat{g}_t = g_t]
\tag{1}
\]

越高越好。需要动作侧车与可恢复相机位姿。后端：`pose_action`。
