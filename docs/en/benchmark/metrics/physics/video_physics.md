# Video Physics Metrics

The `video_physics` group directly compares the generated rollout with the simulator reference in video space. It is a low-level alignment check: a prediction may look visually plausible, but these metrics drop when color, position, shape, or occupied motion regions diverge from the simulator rollout.

## Inputs

| Input | Use |
| --- | --- |
| prediction video | generated rollout from the model |
| `world_gt/rgb.mp4` | simulator-rendered reference RGB rollout |
| `world_gt/mask.npz`, `pred_world/mask.npz` | object or motion-region masks for the IoU metrics |

Some upstream evaluators may fall back to RGB frame differencing to create motion masks, but the current WorldArena core contract still expects predicted world output to provide `mask.npz`.

## Formulas

Let the aligned reference RGB rollout be \(G \in [0,1]^{T \times C \times H \times W}\), and the prediction be \(P \in [0,1]^{T \times C \times H \times W}\). The implementation first truncates both videos to the common frame count and resizes the prediction to the reference resolution; the physics contract separately checks full frame coverage.

### `mse`

\[
\mathrm{mse}
= \frac{1}{TCHW}
  \sum_{t=1}^{T}\sum_{c=1}^{C}\sum_{x \in \Omega}
  \left(P_{t,c,x} - G_{t,c,x}\right)^2
\]

where \(\Omega\) is the \(H \times W\) pixel grid.

### IoU Metrics

Let the aligned reference and prediction masks be:

\[
M^g_t, M^p_t \in \{0,1\}^{H \times W}, \quad t=1,\ldots,T_m
\]

Base IoU is:

\[
\mathrm{IoU}(A,B)=
\begin{cases}
\dfrac{|A \cap B|}{|A \cup B|}, & |A \cup B| > 0 \\
1, & |A \cup B| = 0
\end{cases}
\]

Mean per-frame spatiotemporal IoU:

\[
\mathrm{st\_iou}
= \frac{1}{T_m}\sum_{t=1}^{T_m}
\mathrm{IoU}(M^g_t, M^p_t)
\]

Temporal swept regions:

\[
S^g = \bigvee_{t=1}^{T_m} M^g_t,\qquad
S^p = \bigvee_{t=1}^{T_m} M^p_t
\]

Swept-area IoU:

\[
\mathrm{s\_iou} = \mathrm{IoU}(S^g,S^p)
\]

Reference occupancy weights for weighted swept-area IoU:

\[
w_x = \frac{1}{T_m}\sum_{t=1}^{T_m} M^g_t(x)
\]

If \(\sum_x w_x \le \epsilon\), the implementation sets every \(w_x=1\). Define:

\[
I_x = S^g_x \land S^p_x,\qquad
U_x = S^g_x \lor S^p_x
\]

Then:

\[
\mathrm{ws\_iou}
=
\begin{cases}
\dfrac{\sum_{x \in \Omega} w_x I_x}{\max\left(\sum_{x \in \Omega} w_x U_x,\epsilon\right)}, & \sum_x U_x > 0 \\
1, & \sum_x U_x = 0
\end{cases}
\]

with \(\epsilon=10^{-6}\).

## Metrics

| Metric | Computation | Raw score meaning | Direction | Current normalization |
| --- | --- | --- | --- | --- |
| `mse` | pixelwise mean squared error between aligned prediction RGB and reference RGB | Direct reconstruction error. It penalizes color, texture, shape, spatial position, and timing differences. | lower is better | `[0.0, 0.25]` |
| `st_iou` | IoU between reference and prediction masks at each frame, then averaged | Spatiotemporal occupancy overlap. It checks whether the occupied or moving region is in the right place at each timestep. | higher is better | `[0.0, 1.0]` |
| `s_iou` | IoU after taking the temporal union of reference and prediction masks | Swept-area overlap. It ignores exact timing and checks whether the full spatial area occupied at any time matches. | higher is better | `[0.0, 1.0]` |
| `ws_iou` | swept-area IoU weighted by reference occupancy frequency | Weighted swept-area overlap. It emphasizes pixels that are persistently or frequently occupied in the reference and downweights incidental boundary pixels. | higher is better | `[0.0, 1.0]` |

## Good For

- Checking whether the prediction is close to the simulator reference at the pixel level.
- Checking whether the object or motion region covers the correct spatial area.
- Checking whether the moving subject stays roughly in the right place frame by frame.

## Not Enough For

- `mse` is sensitive to appearance and is not equivalent to physical correctness.
- `s_iou` and `ws_iou` weaken temporal ordering; a trajectory may visit the right places at the wrong times.
- `st_iou` depends on mask quality; missing or misaligned masks reflect artifact quality as well as generation quality.

## Common Failures

- Subject position drift lowers `st_iou`.
- Too-small or too-large swept motion area lowers `s_iou` and `ws_iou`.
- Color, material, or background mismatch increases `mse`.
- Frame-count or coverage violations trigger a contract failure before the metric is trusted.
