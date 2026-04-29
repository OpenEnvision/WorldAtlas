# Physics Metrics & Contracts

All simulator-backed metrics in `config/benchmark_physics.yaml` currently resolve to the `simulator_reference` backend. The benchmark runner computes that backend once per sample and caches the result, so enabling more than one physics metric does not rerun the external evaluator for the same prediction file.

For reporting purposes, configured and optional simulator-backed metrics in the physics track, plus the judge metrics in `config/benchmark_physics_judge.yaml`, all belong to the top-level `physics` dimension.

## Simulator Based Physics Evaluation Metric Groups

`physics_spec.metrics_design` groups required simulator-backed metrics into four groups. Each group's detailed inputs, computation semantics, score direction, and failure behavior live on a dedicated page:

| simulator-backed group | Metrics | Detail page |
| --- | --- | --- |
| `video_physics` | `mse`, `st_iou`, `s_iou`, `ws_iou` | [Video Physics](video_physics.md) |
| `geometry_3d` | `abs_rel`, `rmse`, `delta1`, `delta2`, `delta3` | [Geometry 3D](geometry_3d.md) |
| `temporal_consistency` | `depth_warp_l1`, `depth_warp_charb`, `rgb_warp_charb`, `epe`, `fl_all`, `one_px_out`, plus optional `rgb_warp_lpips` | [Temporal Consistency](temporal_consistency.md) |
| `world_4d` | `chamfer_4d`, `worldline_l2_error`, `worldline_mean_drift`, `worldline_final_drift`, `worldline_fail_rate`, `worldline_length`, plus optional `novel_time_*` | [World 4D](world_4d.md) |

These groups flatten into the current `required_core_metrics` contract carried inside each physics sample. Custom configs that enable optional metrics should also declare matching backend and normalization entries.

## Physics Judge

The VLM-judge physics evaluation is driven by a separate config. It belongs to the top-level `physics` dimension, but it is not one of the simulator-backed artifact groups above:

| Metric family | Metrics | Detail page |
| --- | --- | --- |
| `physics_judge` | `physics_judge_overall`, `physics_judge_phenomenon`, `physics_judge_causality`, `physics_judge_material`, `physics_judge_temporal`, `physics_judge_prompt` | [Physics Judge](physics_judge.md) |

## Normalization Map

The current config declares the following normalization ranges. See the group pages above for per-metric explanations.

WorldArena's generic normalization formula is:

\[
\mathrm{norm}(x)=
\begin{cases}
\mathrm{clip}_{[0,1]}\left(\dfrac{x-l}{u-l}\right), & \text{higher\_is\_better=true}\\[8pt]
\mathrm{clip}_{[0,1]}\left(\dfrac{u-x}{u-l}\right), & \text{higher\_is\_better=false}
\end{cases}
\]

where \(x\) is the raw metric, \(l\) is `lower`, and \(u\) is `upper`. If \(u\approx l\), the implementation directly clips the raw value into \([0,1]\).

| Metric | Source | Better direction | Raw range mapped into `[0, 1]` |
| --- | --- | --- | --- |
| `mse` | `video_physics` | lower is better | `[0.0, 0.25]` |
| `st_iou` | `video_physics` | higher is better | `[0.0, 1.0]` |
| `s_iou` | `video_physics` | higher is better | `[0.0, 1.0]` |
| `ws_iou` | `video_physics` | higher is better | `[0.0, 1.0]` |
| `abs_rel` | `geometry_3d` | lower is better | `[0.0, 1.0]` |
| `rmse` | `geometry_3d` | lower is better | `[0.0, 5.0]` |
| `delta1` | `geometry_3d` | higher is better | `[0.0, 1.0]` |
| `delta2` | `geometry_3d` | higher is better | `[0.0, 1.0]` |
| `delta3` | `geometry_3d` | higher is better | `[0.0, 1.0]` |
| `depth_warp_l1` | `temporal_consistency` | lower is better | `[0.0, 1.0]` |
| `depth_warp_charb` | `temporal_consistency` | lower is better | `[0.0, 1.0]` |
| `rgb_warp_charb` | `temporal_consistency` | lower is better | `[0.0, 0.5]` |
| `epe` | `temporal_consistency` | lower is better | `[0.0, 10.0]` |
| `fl_all` | `temporal_consistency` | lower is better | `[0.0, 1.0]` |
| `one_px_out` | `temporal_consistency` | lower is better | `[0.0, 1.0]` |
| `chamfer_4d` | `world_4d` | lower is better | `[0.0, 1.0]` |
| `worldline_l2_error` | `world_4d` | lower is better | `[0.0, 1.0]` |
| `worldline_mean_drift` | `world_4d` | lower is better | `[0.0, 1.0]` |
| `worldline_final_drift` | `world_4d` | lower is better | `[0.0, 1.0]` |
| `worldline_fail_rate` | `world_4d` | lower is better | `[0.0, 1.0]` |
| `worldline_length` | `world_4d` | higher is better | `[0.0, 1.0]` |
| `physics_judge_*` | `physics_judge` | higher is better | `[0.0, 1.0]` |

## Artifact Contract

Physics evaluation distinguishes between local ground-truth exports and predicted world exports.

### `world_gt/`

| Contract level | Files |
| --- | --- |
| minimum video readiness | `rgb.mp4` |
| full core-metric readiness | `rgb.mp4`, `depth.npz`, `flow.npz`, `mask.npz`, `occlusion.npz`, `cameras.npz`, `points4d.npz`, `worldlines.npz` |
| optional extras | `novel_time_depth.npz`, `novel_time_rgb.npz`, `rgb_frames/`, `export_info.json` |

### `pred_world/`

| Contract level | Files |
| --- | --- |
| required for the current core contract | `depth.npz`, `mask.npz`, `flow.npz`, `cameras.npz` |
| optional extras | `occlusion.npz`, `points4d.npz`, `worldlines.npz`, `novel_time_depth.npz`, `novel_time_rgb.npz` |

The current sample contract sets:

- `require_full_frame_coverage = true`
- `require_pred_world_artifacts = true`

## Success And Failure Semantics

`evaluate_generated_video(...)` marks a sample as `success = true` only when all of the following hold:

- `inspect_physics_case(...)` ends in `status = ready`
- the result contains every metric named in `required_core_metrics`
- frame count and frame coverage satisfy the metric contract
- `pred_world` generation succeeds when `require_pred_world_artifacts = true`

Typical contract failures:

- `prediction frame count does not match the reference rollout`
- `prediction covers only ... of the reference rollout`
- `missing required core physics metrics: ...`
- `predicted world artifacts are required for this physics evaluation`

Those details are exposed in `per_sample_scores.jsonl` under:

- `report_path`
- `world_gt_root`
- `physics_status`
- `pred_world_status`
- `world_gt_status`
- `metric_contract`
- `missing_world_artifacts`
- `missing_required_metrics` when the evaluator omitted a required core metric
