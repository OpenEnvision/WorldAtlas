# Depth Accuracy

`depth_accuracy` measures whether prediction and reference frames induce similar monocular depth fields. It does not use ground-truth depth. Instead, it sends both prediction and reference frames through the same Depth Anything V2 monocular-depth model, then compares whether the estimated relative depth structures match.

This metric focuses on whether the generated result preserves the reference's rough front-back ordering, near/far relationships, and coarse geometric layout.

## What It Measures

- relative depth consistency between prediction and reference
- geometric similarity through a monocular-depth proxy
- whether the generated scene preserves rough front-back structure
- not metric-depth ground truth, not camera-trajectory accuracy, and not 3D reconstruction accuracy

## Principle

Monocular-depth models often capture useful relative depth relationships: foreground objects should appear closer, backgrounds farther away, and layered scene structure should remain coherent. `depth_accuracy` uses this property by converting aligned prediction and reference frames into depth maps and comparing their depth-map shapes.

Because absolute scale from monocular depth is unreliable, WorldArena does not directly compare raw depth values. It first rescales the prediction depth to the reference depth using a median-ratio alignment, then computes AbsRel error. This reduces global-scale bias and makes the metric focus more on whether relative depth structure is similar.

## Current Algorithm

1. WorldArena pairs sampled prediction and reference frames at aligned time indices.
2. If resolutions differ, it resizes the prediction frame to the reference resolution.
3. Both sides are passed through the local Depth-Anything checkpoint resolved from `depth-anything/Depth-Anything-V2-Small-hf` under `ckpt/`.
4. The depth backend outputs `predicted_depth` and bicubic-resizes it back to the original frame size.
5. For each paired frame, the prediction depth map is median-scale aligned to the reference depth map.
6. WorldArena computes frame-level `AbsRel = mean(|d_pred - d_ref| / d_ref)`.
7. The mean is computed only over valid reference-depth pixels above `1e-3`; if there is no valid region, it falls back to the full frame.
8. The raw metric is the mean frame `AbsRel`, so lower is better.

## Error Formula

For the \(i\)-th pair of prediction and reference frames, let Depth Anything V2 produce depth maps \(D_i^{pred}\) and \(D_i^{ref}\). The median-scale alignment factor is:

!!! info "Scale alignment"

    \[
    \alpha_i
    =
    \frac{
        \operatorname{median}(D_i^{ref})
    }{
        \max(\operatorname{median}(D_i^{pred}), 10^{-6})
    }
    \]

The aligned prediction depth is:

!!! info "Aligned depth"

    \[
    \tilde{D}_i^{pred}
    =
    \alpha_i D_i^{pred}
    \]

The valid pixel set is:

!!! info "Valid pixel set"

    \[
    \Omega_i
    =
    \{(x,y)\mid D_i^{ref}(x,y)>10^{-3}\}
    \]

Frame-level AbsRel error is:

!!! info "Frame-level AbsRel"

    \[
    e_i
    =
    \frac{1}{|\Omega_i|}
    \sum_{(x,y)\in\Omega_i}
    \frac{
        \left|
            \tilde{D}_i^{pred}(x,y)-D_i^{ref}(x,y)
        \right|
    }{
        \max(D_i^{ref}(x,y),10^{-6})
    }
    \]

The final sample-level depth error averages all paired frames:

!!! info "Final error"

    \[
    e_{\mathrm{depth}}
    =
    \frac{1}{N}
    \sum_{i=1}^{N}
    e_i
    \]

Here \(N\) is the number of successfully paired frames. Lower error means the prediction and reference have more similar relative depth structure; normalization maps lower error to a higher diagnostic score.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | sampled reference frames |
| Annotation input | none |
| Raw output | one scalar mean `AbsRel` error |
| Normalized output | reverse-scaled diagnostic score from `config/benchmark.yaml`; higher is better |
| Details payload | `model_name`, `frame_absrel`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the depth backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it checks geometry in a way that plain appearance metrics do not.
- Why it is limited: both depth fields are estimated, so this is not ground-truth depth evaluation.
- It is less sensitive to global scale because prediction depth is median-aligned to reference depth before AbsRel.
- Recommended use: use it as a geometry-side appendix metric alongside `reprojection_error` and `depth_collision`.
