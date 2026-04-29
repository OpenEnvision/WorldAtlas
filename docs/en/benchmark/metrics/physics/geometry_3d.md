# Geometry 3D Metrics

The `geometry_3d` group compares predicted-world depth with simulator-exported depth. It checks whether the generated video implies plausible 3D geometry, not just whether RGB frames look similar.

## Inputs

| Input | Use |
| --- | --- |
| `world_gt/depth.npz` | simulator reference depth |
| `pred_world/depth.npz` | prediction depth from the generated video or predicted-world pipeline |
| `world_gt/mask.npz` | optional valid-region constraint for computing depth errors only on target or valid regions |

All metrics are computed only on finite, positive depth locations that pass the valid mask. If no valid pixels remain, the backend omits these metrics, which then triggers required metric failure.

## Formulas

Let reference depth be \(D^g\) and prediction depth be \(D^p\). They are first truncated to the common temporal length. The valid set is:

\[
V =
\left\{(t,x):
\mathrm{finite}(D^g_{t,x}) \land
\mathrm{finite}(D^p_{t,x}) \land
D^g_{t,x}>\epsilon \land
D^p_{t,x}>\epsilon \land
A_{t,x}
\right\}
\]

where \(A_{t,x}\) is the optional valid mask; without an extra mask, \(A_{t,x}=1\). Let \(N=|V|\) and \(\epsilon=10^{-6}\). If \(N=0\), the backend returns no depth metrics.

### `abs_rel`

\[
\mathrm{abs\_rel}
= \frac{1}{N}\sum_{(t,x)\in V}
\frac{|D^p_{t,x}-D^g_{t,x}|}{\max(D^g_{t,x},\epsilon)}
\]

### `rmse`

\[
\mathrm{rmse}
=
\sqrt{
\frac{1}{N}\sum_{(t,x)\in V}
\left(D^p_{t,x}-D^g_{t,x}\right)^2
}
\]

### `delta1` / `delta2` / `delta3`

Define the relative ratio error:

\[
r_{t,x}
=
\max\left(
\frac{D^p_{t,x}}{\max(D^g_{t,x},\epsilon)},
\frac{D^g_{t,x}}{\max(D^p_{t,x},\epsilon)}
\right)
\]

For \(k \in \{1,2,3\}\):

\[
\delta_k
=
\frac{1}{N}
\sum_{(t,x)\in V}
\mathbf{1}\left[r_{t,x}<1.25^k\right]
\]

The metric fields are:

\[
\mathrm{delta1}=\delta_1,\qquad
\mathrm{delta2}=\delta_2,\qquad
\mathrm{delta3}=\delta_3
\]

## Metrics

| Metric | Computation | Raw score meaning | Direction | Current normalization |
| --- | --- | --- | --- | --- |
| `abs_rel` | `mean(abs(pred_depth - gt_depth) / gt_depth)` | Relative depth error. Scale mistakes on nearer objects are amplified. | lower is better | `[0.0, 1.0]` |
| `rmse` | `sqrt(mean((pred_depth - gt_depth)^2))` | Absolute root mean squared depth error. A small number of large geometry mistakes can dominate it. | lower is better | `[0.0, 5.0]` |
| `delta1` | valid-pixel fraction where `max(pred/gt, gt/pred) < 1.25` | Strict depth accuracy, roughly the share of pixels within 25% relative error. | higher is better | `[0.0, 1.0]` |
| `delta2` | valid-pixel fraction where `max(pred/gt, gt/pred) < 1.25^2` | Medium-tolerance depth accuracy. | higher is better | `[0.0, 1.0]` |
| `delta3` | valid-pixel fraction where `max(pred/gt, gt/pred) < 1.25^3` | Loose depth accuracy, mainly exposing broad scale collapse. | higher is better | `[0.0, 1.0]` |

## Good For

- Checking whether subjects and scene elements preserve correct relative depth.
- Detecting large geometric scale errors.
- Validating whether predicted depth can support later 4D and worldline metrics.

## Not Enough For

- Accurate depth does not guarantee RGB appearance, object identity, or causal motion correctness.
- Scores depend on the quality of the predicted-world depth backend.
- Camera or scale convention mismatches can systematically inflate `abs_rel` and `rmse`.

## Common Failures

- The subject is globally too near or too far, increasing `abs_rel` and `rmse`.
- Local geometry cracks or depth holes reduce `delta1/2/3`.
- Missing depth artifacts or empty valid masks cause required metrics to be missing.
