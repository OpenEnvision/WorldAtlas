# Keystroke Action Score

**keystroke_action_score** 为**综合动作控制得分**，对严格准确率、部分准确率与轨迹得分取均值：

\[
S_{\mathrm{action}} = \frac{\mathrm{Acc}_{\mathrm{strict}} + \mathrm{Acc}_{\mathrm{partial}} + S_{\mathrm{traj}}}{3}
\tag{1}
\]

越高越好。需要动作侧车与可恢复相机位姿。后端：`pose_action`。原子构成项：

- [Keystroke Strict Action Accuracy](keystroke_strict_action_accuracy.md)
- [Keystroke Partial Action Accuracy](keystroke_partial_action_accuracy.md)
- [Keystroke Trajectory Score](keystroke_traj_score.md)
