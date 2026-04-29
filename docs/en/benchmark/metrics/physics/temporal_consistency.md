# Temporal Consistency Metrics

The `temporal_consistency` group checks adjacent-frame geometry, optical flow, and RGB self-consistency. It asks whether the prediction evolves continuously over time rather than whether each frame is plausible in isolation.

## Inputs

| Input | Use |
| --- | --- |
| `world_gt/depth.npz`, `pred_world/depth.npz` | depth-warp metrics |
| `world_gt/flow.npz`, `pred_world/flow.npz` | flow error and warping |
| `world_gt/occlusion.npz`, `world_gt/mask.npz` | optional valid regions for excluding occluded or invalid flow areas |
| prediction video | `rgb_warp_charb` and optional `rgb_warp_lpips` |

## Formulas

The implementation uses bilinear sampling for warping. Given a frame tensor \(X_t\) and flow \(F_t(x)=(u_t(x), v_t(x))\), warp is:

\[
W(X_t,F_t)(x)=X_t(x+F_t(x))
\]

Out-of-bounds samples use border padding. The Charbonnier penalty is:

\[
\rho_\epsilon(z)=\sqrt{z^2+\epsilon_c^2}, \qquad \epsilon_c=10^{-3}
\]

### `depth_warp_l1` / `depth_warp_charb`

Let reference flow be \(F^g_t\), reference depth be \(D^g_t\), and prediction depth be \(D^p_t\). The implementation uses:

\[
S=\min(T_{F^g}, T_{D^p}-1, T_{D^g}-1)
\]

For each \(t=1,\ldots,S\), the valid pixel set is:

\[
V_t =
\left\{x:
\mathrm{finite}(W(D^p_t,F^g_t)(x)) \land
\mathrm{finite}(D^g_{t+1,x}) \land
D^g_{t+1,x}>\epsilon \land A_{t,x}
\right\}
\]

where \(A_{t,x}\) comes from non-occluded `world_gt/occlusion.npz` regions or adjacent-mask intersections; if neither is available, \(A_{t,x}=1\). Define:

\[
e_{t,x}=W(D^p_t,F^g_t)(x)-D^g_{t+1,x}
\]

The implementation averages per timestep first, then averages over timesteps with at least one valid pixel. Let that set be \(\mathcal{T}\):

\[
\mathrm{depth\_warp\_l1}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{1}{|V_t|}
\sum_{x\in V_t}|e_{t,x}|
\]

\[
\mathrm{depth\_warp\_charb}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{1}{|V_t|}
\sum_{x\in V_t}\rho_\epsilon(e_{t,x})
\]

### `rgb_warp_charb`

Let prediction RGB be \(P_t \in [0,1]^{C\times H\times W}\), and prediction flow be \(F^p_t\). For:

\[
S=\min(T_{F^p}, T_P-1)
\]

define:

\[
E_{t,c,x}=W(P_t,F^p_t)_{c,x}-P_{t+1,c,x}
\]

If a valid mask \(B_t\) exists, it is expanded across channels; otherwise \(B_t=\Omega\). Timesteps with \(B_t=\varnothing\) are skipped. Let the remaining timesteps be \(\mathcal{T}\):

\[
\mathrm{rgb\_warp\_charb}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{1}{C|B_t|}
\sum_{c=1}^{C}\sum_{x\in B_t}
\rho_\epsilon(E_{t,c,x})
\]

### `epe` / `fl_all` / `one_px_out`

Let reference and prediction flow be \(F^g_{t,x}, F^p_{t,x}\in\mathbb{R}^2\). The valid set is:

\[
V=\{(t,x): \mathrm{finite}(F^g_{t,x}) \land \mathrm{finite}(F^p_{t,x}) \land A_{t,x}\}
\]

Endpoint error and reference flow magnitude:

\[
e_{t,x}=\|F^p_{t,x}-F^g_{t,x}\|_2,\qquad
m_{t,x}=\|F^g_{t,x}\|_2
\]

\[
\mathrm{epe}
=
\frac{1}{|V|}
\sum_{(t,x)\in V} e_{t,x}
\]

Outlier indicator:

\[
o_{t,x}
=
\mathbf{1}
\left[
e_{t,x}>3.0
\land
\frac{e_{t,x}}{\max(m_{t,x},\epsilon)}>0.05
\right]
\]

The current evaluator returns raw percentages:

\[
\mathrm{fl\_all}
=
100\cdot
\frac{1}{|V|}
\sum_{(t,x)\in V} o_{t,x}
\]

\[
\mathrm{one\_px\_out}
=
100\cdot
\frac{1}{|V|}
\sum_{(t,x)\in V}
\mathbf{1}[e_{t,x}>1.0]
\]

## Core Metrics

| Metric | Computation | Raw score meaning | Direction | Current normalization |
| --- | --- | --- | --- | --- |
| `depth_warp_l1` | warp `pred_depth[t]` to the next frame with reference flow, then compare with `gt_depth[t+1]` using L1 error | Whether predicted depth evolves to the correct next-frame geometry under the simulator motion field. | lower is better | `[0.0, 1.0]` |
| `depth_warp_charb` | same alignment as `depth_warp_l1`, using a Charbonnier penalty | A more outlier-robust cross-frame depth warp error. | lower is better | `[0.0, 1.0]` |
| `rgb_warp_charb` | warp prediction RGB frame `t` with prediction flow and compare with prediction RGB frame `t+1` | Self-consistency between the predicted video and predicted flow; it does not directly compare reference RGB. | lower is better | `[0.0, 0.5]` |
| `epe` | endpoint error between `pred_flow` and `gt_flow` | Mean optical-flow vector error in pixels. | lower is better | `[0.0, 10.0]` |
| `fl_all` | fraction or percentage of valid flow pixels satisfying the flow-outlier condition | KITTI-style flow outlier rate, usually requiring EPE greater than 3 px and relative error greater than 5%. | lower is better | `[0.0, 1.0]` |
| `one_px_out` | fraction or percentage of valid flow pixels with EPE greater than 1 px | A stricter one-pixel outlier rate exposing small-motion and boundary instability. | lower is better | `[0.0, 1.0]` |

The current backend evaluator returns raw percentages for `fl_all` and `one_px_out`. The current WorldArena normalization table still declares `[0, 1]`; custom evaluators or configs must keep the raw convention and normalization consistent.

## Optional Metrics

| Metric | Computation | Raw score meaning | Direction |
| --- | --- | --- | --- |
| `rgb_warp_lpips` | same prediction-flow warp target as `rgb_warp_charb`, but using LPIPS perceptual distance | Perceptual temporal self-consistency inside the prediction. | lower is better |

\[
\mathrm{rgb\_warp\_lpips}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\mathrm{LPIPS}\left(
W(P_t,F^p_t)\odot B_t,\,
P_{t+1}\odot B_t
\right)
\]

Enabling `rgb_warp_lpips` requires the LPIPS runtime dependency and a custom normalization entry.

## Good For

- Checking whether adjacent-frame depth and flow are continuous.
- Checking whether the predicted motion field explains the prediction's own RGB evolution.
- Detecting instability around motion boundaries, occlusions, and fast motion.

## Not Enough For

- `rgb_warp_charb` is prediction self-consistency; flow and RGB can be wrong together and still look consistent.
- Flow metrics depend on flow artifact quality.
- This group does not directly judge prompt adherence.

## Common Failures

- Teleportation or skipped motion increases depth-warp error and flow outliers.
- Prediction flow does not match RGB motion, increasing `rgb_warp_charb`.
- Occlusion handling is inconsistent, increasing boundary EPE or one-pixel outliers.
