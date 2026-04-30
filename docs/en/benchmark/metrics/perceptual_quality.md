# Perceptual Quality

`perceptual_quality` measures the visual quality of the prediction itself, without comparing it to the reference rollout.

## What It Measures

- frame sharpness, cleanliness, and aesthetic plausibility
- anchor-frame image quality, not semantic correctness
- prediction quality without any reference image or reference video

## Detailed Algorithm

1. WorldArena evaluates prediction anchor frames only.
2. The reference runtime loads two IQA components:
    - `clipiqa+`
    - `laion_aes`
3. For each anchor frame `f_i`, both component models produce native scores:
    - `clipiqa+` is scaled by `1.0`
    - `laion_aes` is scaled by `10.0`
4. Each component is converted into a bounded score with `component_score = clamp(native / scale, 0, 1)`.
5. The frame score is the arithmetic mean of the required component scores.
6. The raw metric is `raw = mean_i frame_score_i`.
7. Current benchmark normalization is identity because the config already expects `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | prediction anchor frames |
| Reference input | none |
| Annotation input | none |
| Raw output | a scalar mean frame-quality score in `[0, 1]` |
| Normalized output | same scalar in `[0, 1]` |
| Details payload | `frame_scores`, `required_components`, and per-component native and normalized scores |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if any required IQA component is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | always `official` when a numeric score is produced |
| Publication verdict | `Main-table ready` |

- Why it is relatively strong: the algorithm is implemented in-repo, the component stack is explicit, and the aggregation logic is straightforward.
- Why it still is not enough on its own: it does not know whether the content is correct. A beautiful but wrong sample can still score highly.
- Recommended use: acceptable for the main perceptual-quality dimension, but pair it with semantic and control metrics before drawing model-level conclusions.

## Reference

`clipiqa+` follows this work:

```bibtex
@inproceedings{wang2022exploring,
    author = {Wang, Jianyi and Chan, Kelvin CK and Loy, Chen Change},
    title = {Exploring CLIP for Assessing the Look and Feel of Images},
    booktitle = {AAAI},
    year = {2023}
}
```
