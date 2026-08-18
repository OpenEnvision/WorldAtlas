# Metrics Overview

This page follows the public benchmark surface declared in `config/benchmark.yaml`. The four formal suites are the public benchmark surface; `experimental` is a supplemental slice and should not be merged into a single headline benchmark total.

## Reading Rules

The report contract uses the following fields:

- **`raw`** — backend-native value; may be a scalar or a component dictionary
- **`normalized`** — `[0, 1]` score written into aggregation; higher is always better
- **`official`** — enters `per_metric_summary.json` and `per_dimension_summary.json`
- **`diagnostic`** — non-official; never enters headline aggregation
- **`not_applicable`** — strong supervision is absent, so no score is computed or aggregated; recorded in `not_applicable_summary.json`
- **`normalized = null`** — runtime failure or missing input; it is not automatically treated as zero

Per-metric normalization, suite coverage, and official eligibility are documented on each metric page.

## Suite Vectors

| Suite | Metrics | Notes |
| --- | ---: | --- |
| `image_static` | 43 | Image-conditioned static world generation; video predictions are evaluated with the video metric surface when inputs allow |
| `image_dynamic` | 33 | Same as `image_static`, usually with broader motion-prompt coverage |
| `video_static` | 37 | Video-conditioned static world generation |
| `video_dynamic` | 44 | Video-conditioned dynamic world generation |
| `experimental` | 3 | Supplemental slice; not part of the formal headline benchmark |

Whether a metric is official, diagnostic, or `not_applicable` in a given suite depends on supervision and backend success; see each metric page and the [Benchmark Overview](../index.md).

## Six Evaluation Pillars

The Metrics navigation is organized by six top-level dimensions. Each dimension has its own index page with metric families and interpretation notes.

| Dimension | Index page |
| --- | --- |
| Long sequence (memory) | [Long Sequence (Memory)](dimensions/long_sequence.md) |
| Action control | [Action Control](dimensions/action_control.md) |
| 3D/4D consistency | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| Physics | [Physics](dimensions/physics.md) |
| Quality | [Quality](dimensions/quality.md) |
| Real time | [Real Time](dimensions/real_time.md) |
