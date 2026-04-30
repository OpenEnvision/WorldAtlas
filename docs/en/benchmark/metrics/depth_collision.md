# Depth Collision

`depth_collision` uses Video Depth Anything [[Chen et al., 2025](#reference)] metric depth to flag signs that the generated camera path has moved too close to foreground geometry. It focuses on whether the center region in front of the camera becomes abnormally near during a rollout, exposing failures such as wall penetration, object collision, or face-close camera motion.

## What It Measures

- whether the center of the camera view suddenly approaches foreground geometry or obstacles
- camera-path failures such as wall penetration, object collision, and face-close forward motion
- abnormally small estimated depth in the front-facing region during a rollout
- not semantic correctness, true depth accuracy, or physical contact correctness

## Principle

When generated camera motion is unstable, a common failure is that the camera path enters an object, approaches a wall, passes through foreground geometry, or suddenly moves into an implausibly near region. These errors may not be visible from single-frame image quality, but they often appear as unusually small estimated depth in the center of the frame.

`depth_collision` therefore uses Video Depth Anything in metric-depth mode to estimate depth for each frame. Video Depth Anything is designed for consistent depth estimation on long videos, which makes it suitable for producing a relatively stable per-frame depth signal. The metric then checks a fixed-ratio center crop, because the camera's forward direction usually aligns with the image center and the center region best reflects whether an obstacle appears directly in front of the camera.

If any frame has center-crop mean depth below the threshold, the rollout is flagged as a depth collision. This collision means a near-depth warning under an estimated-depth proxy; it is not a collision event derived from 3D colliders, a physics engine, or ground-truth contact labels.

## Current Algorithm

1. `Video-Depth-Anything` [[Chen et al., 2025](#reference)] predicts metric depth for each frame.
2. The metric reads sampled benchmark frames by default; `frame_source = full_video` reads the full prediction.
3. It crops a fixed-ratio center window from each depth map, default `center_crop_ratio = 0.2`.
4. It computes center-window mean metric depth: `center_depth_t = mean(depth_t[center_crop])`.
5. If any frame has `center_depth_t < depth_threshold`, with default threshold `1.0`, the metric flags a collision.
6. The raw score is binary: `raw = 1.0` means collision detected, while `raw = 0.0` means no collision detected.
7. Normalization is reversed: `raw = 0` maps to `normalized = 1`, and `raw = 1` maps to `normalized = 0`.

## Error Formula

For frame \(t\), let Video Depth Anything predict depth map \(D_t\), and let the center crop be \(\mathcal{C}\). The center-region mean depth is:

!!! info "Center depth"

    \[
    \bar{d}_t
    =
    \frac{1}{|\mathcal{C}|}
    \sum_{(x,y)\in\mathcal{C}}
    D_t(x,y)
    \]

Given depth threshold \(\tau\), the depth-collision flag for one video is:

!!! info "Collision decision"

    \[
    e_{\mathrm{collision}}
    =
    \mathbb{1}
    \left[
        \min_t \bar{d}_t < \tau
    \right]
    \]

Here \(\mathbb{1}[\cdot]\) is the indicator function. If any frame has center-region mean depth below the threshold, \(e_{\mathrm{collision}}=1\), indicating a near-depth collision risk; otherwise \(e_{\mathrm{collision}}=0\). The current defaults are \(\tau=1.0\) and `center_crop_ratio = 0.2`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled or full prediction video frames |
| Reference input | unused |
| Annotation input | unused |
| Backend | `video_depth_anything` |
| Default runtime | `repo_root = thirdparty/Video-Depth-Anything`, `checkpoint_path = ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth`, `encoder = vits`, `input_size = 256` |
| Raw output | `0.0` / `1.0` collision flag; lower is better |
| Normalized output | `[0, 1]`, higher is better |
| Detail fields | `center_crop_mean_depths`, `collision`, `first_collision_frame`, `threshold_results`, `depth_shape`, `frame_source`, `frame_indices` |

## Aggregation

| Item | Status |
| --- | --- |
| Suites | `image_static`, `image_dynamic`, `video_static`, `video_dynamic` |
| Aggregation | always `diagnostic` |
| Dimension | `consistency_3d_4d` |

Key limit: a high score only means the current center-crop depth threshold did not fire. It does not prove the camera path is physically valid.

## Reference {#reference}

!!! quote "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"

    Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang.<br>
    arXiv:2501.12375, 2025.

??? note "BibTeX"

    ```bibtex
    @article{video_depth_anything,
        title = {Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
        author = {Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi},
        journal = {arXiv:2501.12375},
        year = {2025},
    }
    ```
