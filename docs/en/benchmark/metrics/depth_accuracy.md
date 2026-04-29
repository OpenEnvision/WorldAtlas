# Depth Accuracy

`depth_accuracy` measures whether prediction and reference induce similar monocular depth fields.

## What It Measures

- agreement between prediction and reference in relative scene depth
- geometry similarity through a monocular-depth proxy
- whether the generated scene preserves rough front-back structure

## Detailed Algorithm

1. WorldArena pairs sampled prediction and reference frames at aligned time indices.
2. The prediction frame is resized to the reference frame resolution if needed.
3. Both sides are passed through the local Depth-Anything checkpoint resolved from `depth-anything/Depth-Anything-V2-Small-hf` under `ckpt/`.
4. For each paired frame, the prediction depth map is median-scale aligned to the reference depth map.
5. WorldArena computes frame-level `AbsRel = mean(|d_pred - d_ref| / d_ref)`.
6. The raw metric is the mean frame `AbsRel`, so lower is better.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | sampled reference frames |
| Annotation input | none |
| Raw output | one scalar mean `AbsRel` error |
| Normalized output | min/max scaled diagnostic score from `config/benchmark.yaml` |
| Details payload | `model_name`, `frame_absrel`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the depth backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it checks geometry in a way that plain appearance metrics do not.
- Why it is limited: both depth fields are estimated, so this is not ground-truth depth evaluation.
- Recommended use: use it as a geometry-side appendix metric alongside `reprojection_error`.
