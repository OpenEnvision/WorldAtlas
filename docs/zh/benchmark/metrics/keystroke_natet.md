# Keystroke nATEt

**keystroke_natet** 报告恒定动作片段上的**归一化平移绝对轨迹误差**（nATEt），为 [Keystroke Trajectory Score](keystroke_traj_score.md) 的越低越好配套项。

\[
\mathrm{nATE}_t = \mathrm{clamp}\!\left(\frac{\mathrm{ATE}_t}{\max(L_{\mathrm{path}}, L_{\min})},\, 0,\, 1\right)
\tag{1}
\]

其中 \(\mathrm{ATE}_t\) 为弧长重采样后的平均位置误差，\(L_{\mathrm{path}}\) 为恢复路径长度（默认 \(L_{\min}=0.5\)）。越低越好。需要动作侧车与可恢复相机位姿。后端：`pose_action`。
