# Physics

The `physics` dimension covers physical motion realism and simulator-backed physics metrics. It includes motion-dynamics metrics in the default benchmark and simulator-world metrics in the independent physics track.

## Overview

| Page | Purpose |
| --- | --- |
| [Physics](physics.md) | Dimension entry point: explains how public benchmark motion metrics, simulator-backed physics, and Physics Judge all roll into the `physics` dimension |
| [Simulator-Backed Physics](../physics/index.md) | Scope, config, report namespace, and interpretation boundaries for the simulator-backed physics track |
| [Physics Workflow](../physics/workflow.md) | Manifest construction, prediction layout, run commands, and report outputs |
| [Physics Metrics & Contracts](../physics/metrics.md) | Metric contract, normalization, artifact requirements, and failure semantics |

## Scope

| Layer | Metrics |
| --- | --- |
| reference motion dynamics | [`pmf`](../pmf.md) |
| motion magnitude and smoothness | [`motion_magnitude`](../motion_magnitude.md), [`motion_smoothness`](../motion_smoothness.md) |
| Long-Sequence dynamics diagnostic | [`long_sequence_dynamic_degree`](../long_sequence.md) |
| Simulator Based Physics evaluation metric groups | [Video Physics](../physics/video_physics.md), [Geometry 3D](../physics/geometry_3d.md), [Temporal Consistency](../physics/temporal_consistency.md), [World 4D](../physics/world_4d.md) |
| standalone VLM physics judge | [Physics Judge](../physics/physics_judge.md) |

## Reading Notes

- `pmf`, `motion_magnitude`, and `motion_smoothness` in the default formal suites target video motion dynamics.
- `config/benchmark_physics.yaml` is a simulator-backed physics config and should not be mixed directly with the default formal-suite headline.
- Simulator-backed formulas, artifact contracts, and failure semantics live in the overview pages [Simulator-Backed Physics](../physics/index.md), [Physics Workflow](../physics/workflow.md), and [Physics Metrics & Contracts](../physics/metrics.md).
- [Physics Judge](../physics/physics_judge.md) belongs to the `physics` dimension, but it is not a simulator-backed artifact group; it is driven separately by `config/benchmark_physics_judge.yaml`.
