# Benchmark Overview

WorldArena reports a suite-specific metric vector for controllable world generation. The benchmark contract is intentionally narrow: fixed manifest, fixed evaluation protocol, fixed eligibility rules, and no benchmark-wide total score.

!!! note "Dataset stats live elsewhere"
    Collection counts, annotation coverage, caption distributions, and media statistics belong to [Dataset Analytics](../overview.md). They are intentionally not duplicated in the Benchmark section.

## Benchmark Surface

| Suite | Modality | Role | What it emphasizes |
| --- | --- | --- | --- |
| `image_static` | image | official | scene semantics, style match, perceptual quality |
| `image_dynamic` | image | official | motion impression in still frames, style match, perceptual quality |
| `video_static` | video | official | scene semantics, appearance, camera control, geometry, photometric consistency |
| `video_dynamic` | video | official | scene semantics, appearance, motion control, motion dynamics |
| `auxiliary` | mixed | diagnostic only | appearance similarity to the reference rollout |
| `experimental` | mixed | diagnostic only | appearance similarity plus motion-dynamics probes |

Official suites are the only suites that may contribute headline benchmark numbers. `auxiliary` and `experimental` remain visible, but they never enter official leaderboard output.

## Reporting Contract

- The primary benchmark result is `per_metric_summary.json`: per-suite averages over official metric outputs.
- `leaderboard.csv` is only a flattened export of the same official suite-by-metric vector.
- `diagnostic_metric_summary.json` is intentionally separate. It contains useful signals that are not strong enough for a formal headline claim.
- A configured metric is not automatically a reported metric. It must produce a numeric value with `eligibility_status = official`.
- WorldArena does not define a benchmark-wide scalar and does not define an official single-number suite score.

## Documentation Map

- [Protocol & Eligibility](protocol.md): temporal sampling, normalization, official vs diagnostic rules.
- [Workflow & Outputs](workflow.md): commands, prediction layout, runtime requirements, output files.
- [Metrics Overview](metrics/index.md): metric families, suite coverage, links to per-metric pages.
- [Environment](../environment.md) and [Runtime Setup](../third_party.md): dependency profiles, heavy metric backends, third-party source layout.
