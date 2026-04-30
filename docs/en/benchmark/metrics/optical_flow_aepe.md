# Optical Flow AEPE

`optical_flow_aepe` measures photometric consistency in generated video: whether consecutive frames maintain stable and coherent visual appearance. It estimates forward and backward optical flow between adjacent frames and asks whether a point can return to its original position after a forward-then-backward tracking cycle.

Smaller error means textures, identity, local structure, and visible regions are more stable over short temporal windows. Larger error often indicates texture flicker, identity drift, object disappearance, or abrupt local appearance changes.

## What It Measures

- short-range visual stability between consecutive frames
- whether forward and backward optical flow form a consistent cycle
- whether generated object identity, texture, and local appearance remain stable across adjacent frames
- not motion ground truth and not target motion correctness

## Principle

Given two consecutive frames \(A\) and \(B\), the metric samples points \(p_A\) in a center crop. The center crop reduces noise from boundary regions, occlusions, out-of-frame motion, and optical-flow extrapolation, so the score focuses more on appearance consistency near the main visible region.

Using forward flow \(F_{A\rightarrow B}\), each point is tracked into the next frame:

!!! info "Forward tracking"

    \[
    p_B
    =
    p_A
    +
    F_{A\rightarrow B}(p_A)
    \]

Using backward flow \(F_{B\rightarrow A}\), the same point is tracked back:

!!! info "Backward tracking"

    \[
    p'_A
    =
    p_B
    +
    F_{B\rightarrow A}(p_B)
    \]

If a generated video is visually stable, a local texture or object point tracked from \(A\) to \(B\) and back to \(A\) should end near its original position: \(p'_A \approx p_A\). If textures flicker, identity changes, local structure drifts, or objects disappear, the forward-backward cycle is harder to close and endpoint error increases.

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. For each consecutive frame pair `(I_t, I_{t+1})`, the wrapped third-party metric runs `SEA-RAFT` [[Wang et al., 2024](#reference)] twice:
    - forward flow `f_{t->t+1}`
    - backward flow `f_{t+1->t}`
3. Both flow fields are cropped to a central region using `crop = 30`.
4. For each sampled point \(p_{A,i}\) inside the crop, the metric applies one round trip:
    - track from frame \(A\) to frame \(B\) with \(F_{A\rightarrow B}(p_{A,i})\)
    - sample backward flow \(F_{B\rightarrow A}(p_{B,i})\) at the destination
    - track back to frame \(A\)
    - compare the returned point \(p'_{A,i}\) with the original point \(p_{A,i}\)
5. The frame-pair photometric-consistency error is the Average End-Point Error (AEPE) over all sampled points.
6. The raw metric is the mean AEPE across all adjacent frame pairs.
7. WorldArena normalizes the scalar with `empirical_min = 0.0`, `empirical_max = 1.192`, and `higher_is_better = false`.

## Error Formula

For one adjacent frame pair, assume the center crop contains \(N\) sampled points. Let the \(i\)-th original point be \(p_{A,i}\), and let the forward-backward round trip return \(p'_{A,i}\). The photometric-consistency error for that frame pair is:

!!! info "Core formula"

    \[
    e_{\mathrm{photometric}}
    =
    \frac{1}{N}
    \sum_{i=1}^{N}
    \left\lVert
        p_{A,i}
        -
        p'_{A,i}
    \right\rVert_2
    \]

Here \(N\) is the number of sampled points in the center crop, and \(\lVert\cdot\rVert_2\) is Euclidean distance between two-dimensional pixel positions. The video-level error averages this value over all adjacent frame pairs; model-level photometric consistency is then aggregated over generated videos in the evaluation set.

Higher AEPE means the forward-backward optical-flow cycle is harder to close and the visual appearance is less stable.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | one scalar forward-backward consistency error |
| Normalized output | one scalar in `[0, 1]`, where higher is better |
| Details payload | `frame_count`, `native_score` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the SEA-RAFT runtime is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Supporting only` |

- Despite the name, this is not classic AEPE against ground-truth flow. It is a forward-backward self-consistency error.
- That makes it useful for temporal artifact analysis, but not sufficient for claiming that the model produced the correct motion.
- Recommended use: keep it as a supporting consistency metric only.

## Reference {#reference}

!!! quote "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow"

    Yihan Wang, Lahav Lipson, and Jia Deng.<br>
    European Conference on Computer Vision (ECCV), 2024.

??? note "BibTeX"

    ```bibtex
    @inproceedings{wang2024sea,
        title = {Sea-raft: Simple, efficient, accurate raft for optical flow},
        author = {Wang, Yihan and Lipson, Lahav and Deng, Jia},
        booktitle = {European Conference on Computer Vision},
        pages = {36--54},
        year = {2024},
        organization = {Springer},
    }
    ```
