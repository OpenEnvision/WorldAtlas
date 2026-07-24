# Keystroke Partial Action Accuracy

**keystroke_partial_action_accuracy** measures **non-empty intersection** between recovered and target atomic action sets at each frame. It tolerates partial compliance when a model activates a subset of the prescribed keys.

Let \(P(\cdot)\) decompose an action into atomic parts (forward, backward, yaw, pitch, etc.). Over \(T\) aligned frames:

\[
\mathrm{Acc}_{\mathrm{partial}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[P(\hat{g}_t) \cap P(g_t) \neq \emptyset]
\tag{1}
\]

The no-op/no-op case counts as a valid partial match. Higher is better. Requires an action sidecar plus recoverable camera poses. Backend: `pose_action`.
