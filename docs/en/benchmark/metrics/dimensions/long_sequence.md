# Long Sequence (Memory)

The `long_sequence` dimension covers state memory, appearance preservation, and long-rollout stability. It asks whether the model preserves seen regions, object presence, revisit behavior, and long-horizon motion or quality over time.

## Scope

| Layer | Metrics |
| --- | --- |
| revisit stability | [`memory_revisit_consistency`](../memory_revisit_consistency.md) |
| seen regions and object persistence | [`seen_region_preservation`](../seen_region_preservation.md), [`object_permanence`](../object_permanence.md) |
| Long-Sequence backend diagnostics | [`long_sequence_motion_smoothness`](../long_sequence.md) |

## Reading Notes

- These metrics mainly answer whether the model remembers what it has already generated.
- `seen_region_preservation` and `object_permanence` are strongest when reference video and masks are available; otherwise they are usually `not_applicable` or diagnostic.
- Bundled Long-Sequence metrics are diagnostic and should be read alongside formal-suite metrics, not as a standalone headline score.
