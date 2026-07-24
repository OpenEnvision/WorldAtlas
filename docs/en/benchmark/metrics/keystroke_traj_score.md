# Keystroke Trajectory Score

**keystroke_traj_score** scores **geometric agreement** between recovered and ideal trajectories on constant-action segments. For each segment, predicted and ideal poses are resampled by arc length; mean translational and rotational errors are normalized by path length and total rotation.

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

Higher is better. Requires an action sidecar plus recoverable camera poses. Backend: `pose_action`. Companion error terms: [Keystroke nATEt](keystroke_natet.md), [Keystroke nATEr](keystroke_nater.md).
