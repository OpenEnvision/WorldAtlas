# Keystroke nATEt

**keystroke_natet** reports the **normalized translational absolute trajectory error** (nATEt) on constant-action segments. It is the lower-is-better companion to [Keystroke Trajectory Score](keystroke_traj_score.md).

\[
\mathrm{nATE}_t = \mathrm{clamp}\!\left(\frac{\mathrm{ATE}_t}{\max(L_{\mathrm{path}}, L_{\min})},\, 0,\, 1\right)
\tag{1}
\]

where \(\mathrm{ATE}_t\) is mean positional error after arc-length resampling and \(L_{\mathrm{path}}\) is the recovered path length (default \(L_{\min}=0.5\)). Lower is better. Requires an action sidecar plus recoverable camera poses. Backend: `pose_action`.
