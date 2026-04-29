# World 4D Metrics

The `world_4d` group compares predicted world structure with simulator world structure in 4D spacetime. It is stricter than single-frame depth because geometry points and worldline trajectories must remain plausible over time.

## Inputs

| Input | Use |
| --- | --- |
| `world_gt/points4d.npz`, `pred_world/points4d.npz` | direct 4D point sets for `chamfer_4d` |
| `world_gt/worldlines.npz`, `pred_world/worldlines.npz` | direct worldline trajectories for worldline metrics |
| `depth.npz`, `cameras.npz`, `mask.npz` | when point sets or prediction worldlines are missing, the backend may lift or reconstruct substitutes from depth, cameras, and masks |
| `novel_time_depth.npz`, `novel_time_rgb.npz` | optional novel-time reconstruction metrics |

## Formulas

### 4D Point Lifting

If `points4d.npz` is missing, the backend can lift sampled points from depth, camera, and mask. For pixel \(x=(u,v)\), depth \(z=D_t(u,v)\), and intrinsics \((f_x,f_y,c_x,c_y)\):

\[
p^{cam}_{t,u,v}
=
\begin{bmatrix}
\frac{u-c_x}{\max(f_x,\epsilon)}z \\
\frac{v-c_y}{\max(f_y,\epsilon)}z \\
z \\
1
\end{bmatrix}
\]

Given camera-to-world matrix \(C^{2w}_t\):

\[
p^{world}_{t,u,v}
=
C^{2w}_t p^{cam}_{t,u,v}
\]

The 4D point is:

\[
q_{t,u,v}
=
\left(
p^{world}_{x},
p^{world}_{y},
p^{world}_{z},
\tau_t
\right),
\qquad
\tau_t=
\begin{cases}
0, & T=1\\
\frac{t}{T-1}, & T>1
\end{cases}
\]

The implementation samples with `point_stride` and keeps at most `max_points4d` points.

### `chamfer_4d`

Let the reference 4D point set be \(G=\{g_i\}_{i=1}^{N}\) and the prediction 4D point set be \(P=\{p_j\}_{j=1}^{M}\), with \(g_i,p_j\in\mathbb{R}^4\). The implementation uses non-squared Euclidean distance:

\[
\mathrm{chamfer\_4d}
=
\frac{1}{2}
\left(
\frac{1}{N}\sum_{i=1}^{N}\min_{1\le j\le M}\|g_i-p_j\|_2
+
\frac{1}{M}\sum_{j=1}^{M}\min_{1\le i\le N}\|p_j-g_i\|_2
\right)
\]

If either point set is empty, the backend returns no `chamfer_4d`.

### Worldline Metrics

Let the 3D coordinates for the \(n\)-th reference and prediction worldline at frame \(t\) be:

\[
X^g_{n,t}, X^p_{n,t}\in\mathbb{R}^3
\]

The valid flag is:

\[
V_{n,t}
=
\mathrm{valid}_{n,t}
\land
\mathrm{finite}(X^g_{n,t})
\land
\mathrm{finite}(X^p_{n,t})
\]

If there are no valid points, the backend returns no worldline metrics. Per-point distance:

\[
d_{n,t}=\|X^p_{n,t}-X^g_{n,t}\|_2
\]

\[
\mathrm{worldline\_l2\_error}
=
\frac{\sum_{n,t}V_{n,t}d_{n,t}}{\sum_{n,t}V_{n,t}}
\]

Displacement from the trajectory start:

\[
\Delta^g_{n,t}=X^g_{n,t}-X^g_{n,0},\qquad
\Delta^p_{n,t}=X^p_{n,t}-X^p_{n,0}
\]

Drift:

\[
\delta_{n,t}=\|\Delta^p_{n,t}-\Delta^g_{n,t}\|_2
\]

\[
\mathrm{worldline\_mean\_drift}
=
\frac{\sum_{n,t}V_{n,t}\delta_{n,t}}{\sum_{n,t}V_{n,t}}
\]

For each trajectory, define the valid-frame set:

\[
I_n=\{t:V_{n,t}=1\}
\]

If \(I_n\neq\varnothing\), final drift is:

\[
f_n=\delta_{n,\max I_n}
\]

\[
\mathrm{worldline\_final\_drift}
=
\frac{1}{|\{n:I_n\neq\varnothing\}|}
\sum_{n:I_n\neq\varnothing} f_n
\]

The default failure threshold is \(\theta=0.1\). Empty trajectories count as failures in fail rate:

\[
\beta_n=
\begin{cases}
1, & I_n=\varnothing\\
\mathbf{1}[f_n>\theta], & I_n\neq\varnothing
\end{cases}
\]

\[
\mathrm{worldline\_fail\_rate}
=
100\cdot
\frac{1}{N_{tracks}}
\sum_{n=1}^{N_{tracks}}
\beta_n
\]

Track length is averaged over non-empty tracks:

\[
\mathrm{worldline\_length}
=
100\cdot
\frac{1}{|\{n:I_n\neq\varnothing\}|}
\sum_{n:I_n\neq\varnothing}
\frac{|I_n|}{T}
\]

### Optional Novel-Time Reconstruction

For any optional novel-time artifact \(Y^g,Y^p\), the implementation first truncates to common temporal length and keeps finite entries:

\[
V=\{i:\mathrm{finite}(Y^g_i)\land\mathrm{finite}(Y^p_i)\}
\]

Let \(r_i=Y^p_i-Y^g_i\). Then:

\[
\mathrm{l1}
=
\frac{1}{|V|}\sum_{i\in V}|r_i|
\]

\[
\mathrm{charb}
=
\frac{1}{|V|}\sum_{i\in V}\sqrt{r_i^2+\epsilon_c^2},
\qquad \epsilon_c=10^{-3}
\]

The metric prefix determines the concrete field:

- `novel_time_depth_l1` / `novel_time_depth_charb`
- `novel_time_rgb_l1` / `novel_time_rgb_charb`

## Core Metrics

| Metric | Computation | Raw score meaning | Direction | Current normalization |
| --- | --- | --- | --- | --- |
| `chamfer_4d` | symmetric Chamfer distance between reference and prediction `(x, y, z, t)` point sets | Whether the overall spatiotemporal point cloud covers the same 4D geometry. | lower is better | `[0.0, 1.0]` |
| `worldline_l2_error` | mean 3D Euclidean distance over all valid points on reference and prediction worldlines | Average trajectory position error at each timepoint. | lower is better | `[0.0, 1.0]` |
| `worldline_mean_drift` | compare each trajectory's displacement from its start and average over valid points | Whether trajectory shape and accumulated motion match the reference. | lower is better | `[0.0, 1.0]` |
| `worldline_final_drift` | average drift at each trajectory's last valid point | Whether the rollout endpoint accumulates large trajectory error. | lower is better | `[0.0, 1.0]` |
| `worldline_fail_rate` | share of tracks whose final drift exceeds the evaluator threshold | Trajectory failure rate. | lower is better | `[0.0, 1.0]` |
| `worldline_length` | average valid length or coverage of each worldline across the rollout | Trajectory completeness: whether worldlines stay trackable into later frames. | higher is better | `[0.0, 1.0]` |

The current backend evaluator returns raw percentages for `worldline_fail_rate` and `worldline_length`. The current WorldArena normalization table still declares `[0, 1]`; custom evaluators or configs must keep the raw convention and normalization consistent.

## Optional Metrics

| Metric | Computation | Raw score meaning | Direction |
| --- | --- | --- | --- |
| `novel_time_depth_l1` | L1 reconstruction error between prediction and reference `novel_time_depth` | Depth extrapolation/interpolation error at held-out time samples outside the core rollout. | lower is better |
| `novel_time_depth_charb` | Charbonnier reconstruction error between prediction and reference `novel_time_depth` | More outlier-robust novel-time depth error. | lower is better |
| `novel_time_rgb_l1` | L1 reconstruction error between prediction and reference `novel_time_rgb` | RGB reconstruction error at held-out time samples outside the core rollout. | lower is better |
| `novel_time_rgb_charb` | Charbonnier reconstruction error between prediction and reference `novel_time_rgb` | More outlier-robust novel-time RGB error. | lower is better |

Enabling optional novel-time metrics requires the corresponding artifacts in both `world_gt/` and `pred_world/`, plus custom normalization entries.

## Good For

- Checking whether the prediction forms a continuous, trackable 4D world structure.
- Checking whether object trajectories remain close to simulator reference over time.
- Detecting accumulated drift near the end of long rollouts.

## Not Enough For

- Correct 4D point structure does not guarantee prompt adherence, material behavior, or visual quality.
- Lifting and reconstruction depend on depth, camera, and mask artifact quality.
- `chamfer_4d` is a set distance and does not preserve object identity correspondence by itself.

## Common Failures

- Wrong prediction depth or camera pose shifts lifted points, increasing `chamfer_4d`.
- Worldlines become invalid or are lost mid-rollout, reducing `worldline_length`.
- Trajectory shape is roughly right but scale is wrong, worsening both `worldline_l2_error` and drift metrics.
