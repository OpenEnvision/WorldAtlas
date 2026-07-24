# Keystroke nATEr

**keystroke_nater** reports the **normalized rotational absolute trajectory error** (nATEr) on constant-action segments. It is the lower-is-better companion to [Keystroke Trajectory Score](keystroke_traj_score.md).

\[
\mathrm{nATE}_r = \mathrm{clamp}\!\left(\frac{\mathrm{ATE}_r}{\max(\Phi_{\mathrm{total}}, \Phi_{\min})},\, 0,\, 1\right)
\tag{1}
\]

where \(\mathrm{ATE}_r\) is mean rotational error in degrees after arc-length resampling and \(\Phi_{\mathrm{total}}\) is total recovered rotation (default \(\Phi_{\min}=10°\)). Lower is better. Requires an action sidecar plus recoverable camera poses. Backend: `pose_action`.
