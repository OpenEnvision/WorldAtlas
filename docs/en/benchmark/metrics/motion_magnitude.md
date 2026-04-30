# Motion Magnitude

`motion_magnitude` measures how much motion the prediction contains, regardless of whether that motion is correct. It is mainly used to catch conservative generation: some models preserve visual stability and image quality by producing very small motion, or even nearly static videos, despite prompts that require movement.

## What It Measures

- overall motion amplitude in the prediction
- per-pair optical-flow strength
- whether the video is nearly static or contains observable movement
- not motion correctness and not smoothness; large motion does not imply correct direction, object behavior, or semantics

## Principle

A common video-generation failure mode is conservative generation: the output looks clean, stable, and low-flicker, but objects barely move. Such outputs can score well on static image-quality metrics and can also look smooth because static videos are easy to interpolate.

`motion_magnitude` directly measures internal motion amount. It estimates dense optical flow for every pair of adjacent frames, converts the two-dimensional flow vector at each pixel into a magnitude, takes the median magnitude for each frame pair, and averages those values over the whole video. The median is used instead of a maximum or mean to reduce the impact of a few outlier flow vectors, boundary artifacts, or local glitches.

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. For each consecutive frame pair, the wrapped third-party metric runs `SEA-RAFT` [[Wang et al., 2024](#reference)] to estimate optical flow.
3. It converts the flow into a magnitude map `m = sqrt(u^2 + v^2)`.
4. The pair statistic is the median value of that magnitude map over all pixels.
5. The raw metric is the mean pair statistic across all consecutive frame pairs.
6. WorldArena applies the current z-score calibration:
    - `avg = 3.2425`
    - `std = 3.4505`
    - `z_min = -0.8638`
    - `z_max = 2.7498`
    - output range `[0.25, 0.75]`

## Error Formula

Given two adjacent frames \((I_t, I_{t+1})\), SEA-RAFT estimates the optical-flow field:

!!! info "Optical-flow field"

    \[
    \mathbf{F}_t(x,y)
    =
    (u_t(x,y), v_t(x,y))
    \]

The corresponding flow-magnitude map is:

!!! info "Flow magnitude"

    \[
    F_t(x,y)
    =
    \sqrt{
        u_t(x,y)^2
        +
        v_t(x,y)^2
    }
    \]

The frame-pair motion magnitude is the median flow magnitude over all pixels:

!!! info "Frame-pair motion magnitude"

    \[
    s^{(t)}_{\mathrm{motion\text{-}mag}}
    =
    \operatorname{median}_{(x,y)}
    F_t(x,y)
    \]

The final raw metric is the average over adjacent frame pairs:

!!! info "Final raw score"

    \[
    s_{\mathrm{motion\text{-}mag}}
    =
    \frac{1}{T-1}
    \sum_{t=0}^{T-2}
    s^{(t)}_{\mathrm{motion\text{-}mag}}
    \]

Higher values mean the generated video contains more visible motion. If a model emits a nearly static video, the median flow magnitude is close to zero and the final score is low.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none in the official backend |
| Annotation input | none |
| Raw output | one scalar average motion-magnitude score |
| Normalized output | one scalar in `[0, 1]` after z-score mapping |
| Details payload | `frame_count`, `native_score`, `frame_indices`, `sampled_frame_indices` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the flow backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Main-table usable, with the caveat that it only measures motion amount` |

- This metric is useful for distinguishing nearly static outputs from outputs that actually move.
- It does not answer whether the motion is the right motion. A wildly wrong but energetic video can still score highly.
- It also does not measure whether the motion is smooth; large-motion videos with severe flicker can still score highly.
- Recommended use: treat it as one primary dynamics or motion-amount metric, and interpret it next to `motion_accuracy` and `motion_smoothness`.

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
