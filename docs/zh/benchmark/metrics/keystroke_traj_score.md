# Keystroke Trajectory Score

**keystroke_traj_score** 在恒定动作片段上评估恢复轨迹与理想轨迹的**几何一致**。每段内预测与理想位姿按弧长重采样；平移与旋转平均误差分别由路径长度与总旋转量归一化。

\[
\mathrm{nATE}_t = \mathrm{clamp}\!\left(\frac{\mathrm{ATE}_t}{\max(L_{\mathrm{path}}, L_{\min})},\, 0,\, 1\right),
\quad
\mathrm{nATE}_r = \mathrm{clamp}\!\left(\frac{\mathrm{ATE}_r}{\max(\Phi_{\mathrm{total}}, \Phi_{\min})},\, 0,\, 1\right)
\tag{1}
\]

\[
S_{\mathrm{traj}} = 1 - \frac{\mathrm{nATE}_t + \mathrm{nATE}_r}{2}
\tag{2}
\]

越高越好。需要动作侧车与可恢复相机位姿。后端：`pose_action`。配套误差项见 [Keystroke nATEt](keystroke_natet.md)、[Keystroke nATEr](keystroke_nater.md)。
