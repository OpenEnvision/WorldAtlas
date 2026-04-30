# Trajectory Dynamics Metrics

The `trajectory_dynamics` group evaluates simulator-backed inertia behavior over the full rollout. It is designed for settings where an end frame alone is insufficient: a prediction can land near the correct endpoint while producing non-inertial motion in the middle.

## Inputs

| Input | Use |
| --- | --- |
| `world_gt/worldlines.npz` | reference tracked points, with `xyz`, `valid`, and usually `pixels` |
| `pred_world/worldlines.npz` | prediction tracked points; WorldArena reconstructs this from `pred_world/depth.npz`, `pred_world/cameras.npz`, and reference worldline pixels when it is absent |
| `physics_spec.trajectory_metric.coordinate_space` | optional coordinate convention; default is `world_xyz`, while `normalized_pixels` uses pixel tracks divided by sample width and height |

The default simulator implementation uses `world_xyz` because PSIVG's artifact contract provides simulator worldlines. A custom manifest may set `coordinate_space: normalized_pixels` to match 2D normalized pixel formulas, but then both reference and prediction worldline artifacts must contain `pixels`.

## Formulas

Let \(p^g_{n,t}\) and \(p^p_{n,t}\) be reference and prediction positions for track \(n\) at frame \(t\). The valid mask requires both tracks to be valid and finite.

### `trajectory_rmse`

\[
\mathrm{trajectory\_rmse}
=
\sqrt{
\frac{1}{|V|}
\sum_{(n,t)\in V}
\|p^g_{n,t}-p^p_{n,t}\|_2^2
}
\]

Lower is better.

### `final_position_error`

For each track with a valid final frame, define the reference path length:

\[
L^g_n =
\sum_{t=1}^{T-1}
\|p^g_{n,t+1}-p^g_{n,t}\|_2
\]

Stationary tracks with \(L^g_n \approx 0\) are skipped because the ratio is undefined. The metric averages:

\[
\mathrm{final\_position\_error}
=
\frac{1}{|\mathcal{N}|}
\sum_{n\in\mathcal{N}}
\frac{\|p^g_{n,T}-p^p_{n,T}\|_2}{L^g_n}
\]

Lower is better.

### `speed_similarity`

Velocities are first differences:

\[
v^g_{n,t}=p^g_{n,t}-p^g_{n,t-1},\qquad
v^p_{n,t}=p^p_{n,t}-p^p_{n,t-1}
\]

The score is the mean cosine similarity over valid adjacent pairs:

\[
\mathrm{speed\_similarity}
=
\frac{1}{|\mathcal{V}|}
\sum_{(n,t)\in\mathcal{V}}
\frac{v^g_{n,t}\cdot v^p_{n,t}}
{\|v^g_{n,t}\|_2\|v^p_{n,t}\|_2}
\]

If both velocity vectors are near zero, WorldArena treats that pair as similarity `1`; if only one is near zero, it is `0`.

### `acceleration_similarity`

Accelerations are velocity differences:

\[
a^g_{n,t}=v^g_{n,t}-v^g_{n,t-1},\qquad
a^p_{n,t}=v^p_{n,t}-v^p_{n,t-1}
\]

The metric is the mean cosine similarity over valid acceleration pairs. The same near-zero convention as `speed_similarity` is used, so a correctly constant-velocity trajectory receives a high acceleration score instead of becoming undefined.

### `directional_consistency`

For each track with valid first and final positions:

\[
d^g_n=p^g_{n,T}-p^g_{n,1},\qquad
d^p_n=p^p_{n,T}-p^p_{n,1}
\]

Let \(\theta_n\) be the angle between them in degrees. The per-track score is:

\[
S^{dir}_n=\frac{180-\theta_n}{180}
\]

The metric averages \(S^{dir}_n\) over valid tracks. Larger is better.

## Core Metrics

| Metric | Raw score meaning | Direction | Current normalization |
| --- | --- | --- | --- |
| `trajectory_rmse` | global deviation between predicted and reference tracked-point paths | lower is better | `[0.0, 1.0]` |
| `final_position_error` | final endpoint error divided by reference path length | lower is better | `[0.0, 1.0]` |
| `speed_similarity` | local velocity direction agreement | higher is better | `[-1.0, 1.0]` |
| `acceleration_similarity` | local acceleration pattern agreement | higher is better | `[-1.0, 1.0]` |
| `directional_consistency` | start-to-end direction agreement | higher is better | `[0.0, 1.0]` |

## Failure Modes

- Missing `world_gt/worldlines.npz` or `pred_world/worldlines.npz`.
- Missing `pixels` when `coordinate_space` is set to `normalized_pixels`.
- Mismatched trajectory shapes or no valid tracked points.
- No valid non-stationary track for `final_position_error`.
