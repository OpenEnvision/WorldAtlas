# Style Consistency

`style_consistency` measures how closely the prediction preserves the reference rollout's global style statistics at matched anchor frames.

## What It Measures

- similarity of color palette, texture distribution, and coarse appearance statistics
- reference-conditioned style preservation
- not prompt following, not motion correctness, and not perceptual realism by itself

## Detailed Algorithm

1. WorldArena evaluates paired anchor frames from the reference and prediction. For videos, the pairing follows the current official protocol frames. For images, the pair is the single image.
2. Each frame pair is resized and normalized for a VGG19 feature extractor.
3. Feature activations are captured after VGG19 layer indices `0`, `2`, `5`, `7`, and `10`.
4. For each captured layer `l`, the runtime computes Gram matrices for the reference and prediction features and then the layer distance `d_{i,l} = MSE(Gram(ref_{i,l}), Gram(pred_{i,l}))`.
5. The frame distance is `d_i = mean_l d_{i,l}`.
6. The frame similarity is `s_i = clamp(1 / (1 + d_i), 0, 1)`.
7. The raw metric is `raw = mean_i s_i`. Current benchmark normalization is identity because the output is already bounded in `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | prediction anchor frames |
| Reference input | paired reference anchor frames |
| Annotation input | none |
| Raw output | a scalar mean style similarity in `[0, 1]` |
| Normalized output | same scalar in `[0, 1]` |
| Details payload | `frame_scores`, `native_frame_distances`, `native_layer_distances` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the VGG backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | always `official` when a numeric score is produced |
| Publication verdict | `Main-table ready` |

- Why it is relatively strong: the algorithm is implemented in-repo, the transformation from distance to similarity is explicit, and there is no hidden supervision gate.
- Important reproducibility caveat: the reference runtime uses `512x512` preprocessing on CUDA and `128x128` on CPU. CPU fallback is therefore **not** numerically equivalent to GPU evaluation.
- Important interpretation caveat: this metric can reward close appearance copying even when the sample is semantically wrong or temporally poor.
- Recommended use: acceptable for the main style dimension, but pin the evaluation environment and never present it as a complete quality metric.
