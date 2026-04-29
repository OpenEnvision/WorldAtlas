# Seen Region Preservation

`seen_region_preservation` asks whether scene content that remains visible after the first reference frame is preserved by the prediction.

## What It Measures

- fidelity on regions that were already visible in the first reference frame
- masked prediction-vs-reference agreement on those persistent regions
- not prompt following, not camera-ground-truth accuracy, and not dynamic-object tracking

## Detailed Algorithm

1. The metric requires a reference video with at least two sampled frames.
2. WorldArena uses the first reference frame as the source frame.
3. It optionally loads ignore masks from `sample.mask_path` or `annotation_path/dyn_masks.npz` so that dynamic foreground regions do not dominate the static-scene probe.
4. For each later reference frame, it computes forward and backward Farneback optical flow between the source frame and the target frame.
5. A pixel is treated as "seen" when:
    - the backward-warped coordinate stays inside the image
    - forward/backward flow consistency error is below the current threshold
    - the optional ignore mask does not suppress that location
6. The seen mask is lightly cleaned with morphology before scoring.
7. For each later frame, the prediction is resized to the reference resolution and compared to the reference only on the seen mask.
8. The frame score is the mean of the available masked components:
    - masked PSNR mapped into `[0, 1]`
    - masked DINOv2 similarity on masked crops
9. Frame scores are weighted by mask coverage, and the raw metric is the weighted average across frames.
10. Current normalization is direct bounded scaling in `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction video frames |
| Reference input | sampled reference video frames |
| Annotation input | optional ignore masks from `mask_path` or `dyn_masks.npz` |
| Raw output | one scalar weighted mean score in `[0, 1]` |
| Normalized output | same bounded scalar under the default config |
| Details payload | `frame_scores` with `frame_index`, `coverage`, `psnr_component`, `dino_component`, `score`, plus frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string when no valid seen-region scores can be formed |

## Integration Notes

- Current aggregation: `official` in `image_static`, `image_dynamic`, and `video_static` when a numeric score is produced; pure image references usually return unavailable.
- Small or empty masks are skipped rather than treated as zeros.
- This metric is most informative when read together with [`novel_region_fidelity`](novel_region_fidelity.md), because the two metrics separate "preserve what stayed visible" from "render what became newly visible".
