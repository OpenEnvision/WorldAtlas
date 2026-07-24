# Final State Accuracy

**final_state_accuracy** is a scenario-conditional simulator-backed metric for final-state classification tasks such as lever balance outcomes. It reports whether the predicted final state matches the ground-truth final state.

This metric is configured in `config/benchmark_physics.yaml`, but it is `not_applicable` unless the sample declares a final-state rule or explicit labels in `physics_spec.final_state_metric`.

## Inputs

| Input | Use |
| --- | --- |
| `physics_spec.final_state_metric` | declares explicit labels or a rule for deriving labels from simulator worldlines |
| `world_gt/worldlines.npz` | used to derive the ground-truth label when `true_label` is not provided |
| `pred_world/worldlines.npz` | used to derive the prediction label when `predicted_label` is not provided |

## Supported Contract

Explicit labels can be supplied directly:

```yaml
physics_spec:
  final_state_metric:
    true_label: left_down
    predicted_label: left_down
```

For lever-like cases, labels can be derived from two tracked endpoints:

```yaml
physics_spec:
  final_state_metric:
    type: lever
    left_track_index: 0
    right_track_index: 1
    axis: y
    balanced_tolerance: 0.02
    lower_is_down: true
```

The default labels are `left_down`, `right_down`, and `balanced`. They can be overridden with:

```yaml
labels:
  left_down: left_down
  right_down: right_down
  balanced: balanced
```

## Formula

For each sample:

\[
\mathrm{final\_state\_accuracy}
=
\mathbf{1}\left[\hat{y}=y\right]
\]

The aggregate over a set of \(S\) samples is the mean:

\[
\mathrm{Accuracy}
=
\frac{1}{S}\sum_{i=1}^{S}
\mathbf{1}\left[\hat{y}^{(i)}=y^{(i)}\right]
\]

## Lever Rule

For the lever rule, WorldArena reads the final frame where both endpoints are valid. With `axis: y` and `lower_is_down: true`:

- `left_down` if the left endpoint is lower than the right endpoint by more than `balanced_tolerance`
- `right_down` if the right endpoint is lower than the left endpoint by more than `balanced_tolerance`
- `balanced` otherwise

The same rule is applied to reference and prediction worldlines unless an explicit label is supplied.

## Output

| Field | Meaning |
| --- | --- |
| Raw score | `1.0` for a matched final state, `0.0` otherwise |
| Normalized score | identity mapping over `[0, 1]` |
| Eligibility | official when the sample provides a final-state contract; `not_applicable` otherwise |

## Failure Modes

- The sample declares a final-state metric but omits both explicit labels and a supported rule.
- Lever track indices are missing or outside the worldline array.
- The selected final-state tracks have no jointly valid frame.
