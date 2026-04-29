# Novel Region Fidelity

`novel_region_fidelity` asks whether newly exposed parts of the scene are rendered plausibly when the reference camera reveals content that was not visible in the first frame.

## What It Measures

- fidelity on regions that become newly visible over time
- masked prediction-vs-reference agreement on those newly exposed regions
- not a direct camera-control metric and not a pure hallucination detector

## Detailed Algorithm

1. The metric requires a reference video with at least two sampled frames.
2. WorldArena derives "seen" regions from forward/backward Farneback flow between the first reference frame and each later reference frame.
3. It then defines the "novel" region as the complement of a dilated seen mask, optionally removing dynamic foreground areas supplied by `mask_path` or `dyn_masks.npz`.
4. For each later frame, the prediction is resized to the reference resolution and compared to the reference only on the novel mask.
5. The frame score is the mean of the available masked components:
    - masked PSNR mapped into `[0, 1]`
    - masked DINOv2 similarity on masked crops
6. Frame scores are weighted by novel-region coverage.
7. The raw metric is the weighted average across valid later frames.
8. Current normalization is direct bounded scaling in `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction video frames |
| Reference input | sampled reference video frames |
| Annotation input | optional ignore masks from `mask_path` or `dyn_masks.npz` |
| Raw output | one scalar weighted mean score in `[0, 1]` |
| Normalized output | same bounded scalar under the default config |
| Details payload | `frame_scores` with `frame_index`, `coverage`, `psnr_component`, `dino_component`, `score`, plus frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string when no valid novel-region scores can be formed |

## Integration Notes

- Current aggregation: `official` in `image_static`, `image_dynamic`, and `video_static` when a numeric score is produced; pure image references usually return unavailable.
- The metric is complementary to [`seen_region_preservation`](seen_region_preservation.md): together they split the problem into persistent regions and newly exposed regions.
- Like the seen-region metric, tiny masks are skipped instead of becoming zeros.
