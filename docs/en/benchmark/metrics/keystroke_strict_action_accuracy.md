# Keystroke Strict Action Accuracy

**keystroke_strict_action_accuracy** measures **exact per-frame agreement** between recovered and target discrete keyboard-action labels. Camera poses recovered from the prediction (ViPE sidecar) are discretized into atomic action sets and compared against the ground-truth action sequence from the action payload.

Let \(\hat{g}_t\) and \(g_t\) denote recovered and target action sets at frame \(t\). Over \(T\) aligned frames:

\[
\mathrm{Acc}_{\mathrm{strict}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[\hat{g}_t = g_t]
\tag{1}
\]

Higher is better. Requires an action sidecar plus recoverable camera poses. Backend: `pose_action`.
