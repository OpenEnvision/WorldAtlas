# Keystroke Action Score

**keystroke_action_score** is a **composite action-control score** averaging strict accuracy, partial accuracy, and trajectory score:

\[
S_{\mathrm{action}} = \frac{\mathrm{Acc}_{\mathrm{strict}} + \mathrm{Acc}_{\mathrm{partial}} + S_{\mathrm{traj}}}{3}
\tag{1}
\]

Higher is better. Requires an action sidecar plus recoverable camera poses. Backend: `pose_action`. Atomic constituents:

- [Keystroke Strict Action Accuracy](keystroke_strict_action_accuracy.md)
- [Keystroke Partial Action Accuracy](keystroke_partial_action_accuracy.md)
- [Keystroke Trajectory Score](keystroke_traj_score.md)
