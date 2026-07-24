# Keystroke nATEr

**keystroke_nater** 报告恒定动作片段上的**归一化旋转绝对轨迹误差**（nATEr），为 [Keystroke Trajectory Score](keystroke_traj_score.md) 的越低越好配套项。

\[
\mathrm{nATE}_r = \mathrm{clamp}\!\left(\frac{\mathrm{ATE}_r}{\max(\Phi_{\mathrm{total}}, \Phi_{\min})},\, 0,\, 1\right)
\tag{1}
\]

其中 \(\mathrm{ATE}_r\) 为弧长重采样后的平均旋转误差（度），\(\Phi_{\mathrm{total}}\) 为恢复轨迹总旋转量（默认 \(\Phi_{\min}=10°\)）。越低越好。需要动作侧车与可恢复相机位姿。后端：`pose_action`。
