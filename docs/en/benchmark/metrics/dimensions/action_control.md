# Action Control

The `action_control` dimension covers scene control, camera control, and object/action control. It asks whether the model follows prompts, camera instructions, target-object constraints, and explicit motion supervision.

## Scope

| Layer | Metrics |
| --- | --- |
| prompt / scene control | [`prompt_alignment`](../prompt_alignment.md), [`semantic_alignment`](../semantic_alignment.md) |
| camera control | [`camera_error`](../camera_error.md) |
| object control | [`object_detection`](../object_detection.md) |
| action / motion control | [`motion_accuracy`](../motion_accuracy.md) |

## Reading Notes

- `prompt_alignment` is official only when explicit instruction supervision exists and the sample is not self-conditioned.
- `object_detection` scores only explicit structured object annotations; caption-derived prompts are not scored.
- `camera_error` and `motion_accuracy` require strong sidecars such as pose or dynamic masks; missing supervision is not converted into zero.
