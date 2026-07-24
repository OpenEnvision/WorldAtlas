# Object Permanence

**object_permanence** measures whether a masked dynamic object keeps the appearance trajectory implied by the reference rollout.

## What It Measures

- whether a foreground object stays recognizably the same entity over time
- agreement between the prediction's object-appearance curve and the reference object's appearance curve
- not raw motion amplitude and not whole-frame similarity

## Detailed Algorithm

1. The metric requires a readable reference video and aligned dynamic masks from `sample.mask_path` or `annotation_path/dyn_masks.npz`.
2. WorldArena aligns the mask sequence to the sampled reference frames.
3. It chooses an anchor object crop from the frame just before the sampled window when possible; otherwise it falls back to the first sampled frame.
4. For each sampled timestep:
    - it extracts the masked object crop from the reference frame
    - it resizes the prediction frame to the reference resolution and extracts the masked object crop
    - it encodes the anchor crop, reference crop, and prediction crop with DINOv2 features
5. Two similarities are computed per frame:
    - `gt_similarity`: anchor crop vs reference crop
    - `prediction_similarity`: anchor crop vs prediction crop
6. The per-frame permanence score is `1 - abs(prediction_similarity - gt_similarity)`, clamped into `[0, 1]`.
7. Frame scores are weighted by mask coverage.
8. The raw metric is the weighted average over valid frames.
9. Current normalization is direct bounded scaling in `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction video frames |
| Reference input | sampled reference video frames plus the chosen anchor frame |
| Annotation input | aligned dynamic masks from `mask_path` or `dyn_masks.npz` |
| Raw output | one scalar permanence score in `[0, 1]` |
| Normalized output | same bounded scalar under the default config |
| Details payload | `anchor_frame_index`, `frame_scores`, `gt_similarity_curve`, `prediction_similarity_curve`, plus frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string when masks are unavailable or no valid masked crops can be formed |

## Integration Notes

- Current aggregation: `official` in `image_static`, `image_dynamic`, and `video_dynamic` when a reference video plus aligned masks exist and a numeric score is produced.
- This metric is appearance-persistence oriented. It should be read together with [`motion_accuracy`](motion_accuracy.md) and [`pmf`](pmf.md) if you care about both "does the object stay the same" and "does it move correctly".
- Mask quality matters directly. Poor mask alignment can degrade the score even when the generated object looks plausible.
