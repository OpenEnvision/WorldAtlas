# Reprojection Error

**reprojection_error** measures whether a prediction video is **geometrically self-consistent under the wrapped DROID-SLAM reconstruction path** [[Teed and Deng, 2021](#reference)]. It requires only the prediction itself—no reference video and no camera ground truth. The core question is whether camera poses and dense depth recovered from the generated video can reproject scene points close to their observed pixel locations in other frames.

Unlike sparse reconstruction methods such as COLMAP [[Schönberger and Frahm, 2016](#reference-colmap-sfm), [Schönberger et al., 2016](#reference-colmap-mvs)], which rely mainly on a small set of high-quality feature matches, DROID-SLAM follows a **dense SLAM** path. Its differentiable Dense Bundle Adjustment (DBA) layer jointly refines camera poses and per-pixel depth while keeping them consistent with optical-flow correspondences. Geometric failures in generated video often appear not only near a few matched points, but also as background warping, local object drift, unstable depth layering, or cross-frame structures that fail to close. Dense DBA uses as many valid pixels and co-visible points as possible, making the metric closer to “can the full rollout be explained by one 3D reconstruction?”

After DBA refinement, reprojection error over the valid co-visible set \(\mathcal{V}\) is

\[
e_{\mathrm{reproj}}
=
\frac{1}{|\mathcal{V}|}
\sum_{(i,j)\in\mathcal{V}}
\left\lVert
    p^{*}_{ij}
    -
    \Pi(P_{ij})
\right\rVert_2
\tag{1}
\]

where \(p^{*}_{ij}\) is the observed pixel location in frame \(i\), \(P_{ij}\) is the 3D point recovered from pose and depth, and \(\Pi(\cdot)\) is the projection operator. If the generated video is 3D-consistent, the same scene point should reproject near its observation in other frames; geometric drift, depth instability, or cross-frame inconsistency increases this distance.

In WorldArena, the runtime decodes the full prediction into a frame sequence, runs DROID-SLAM to estimate trajectory and internal geometry, and sets the raw score to `raw = mean(valid_errors)` over accepted reprojection errors. WorldArena reverse-normalizes on the empirical range \([0, 1.0719]\) to \([0, 1]\) (**higher normalized score is better**). If SLAM reconstruction fails or no valid score is produced, it returns `raw = null` and `normalized = null`. When DROID-SLAM succeeds, the metric is **official** in all formal suites and enters the `consistency_3d_4d` aggregate. It is a useful geometry-consistency signal, but it does not replace geometry ground-truth metrics such as camera error, and fixed-intrinsics SLAM assumptions warrant cautious interpretation across heterogeneous data.

## References {#reference}

!!! quote "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Zachary Teed and Jia Deng.  
    Advances in Neural Information Processing Systems (NeurIPS), 2021.

<a id="reference-colmap-sfm"></a>

!!! quote "Structure-from-Motion Revisited"

    Johannes L. Schönberger and Jan-Michael Frahm.  
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

<a id="reference-colmap-mvs"></a>

!!! quote "Pixelwise View Selection for Unstructured Multi-View Stereo"

    Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys.  
    European Conference on Computer Vision (ECCV), 2016.

??? note "BibTeX"

    ```bibtex
    @article{teed2021droid,
        title = {Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras},
        author = {Teed, Zachary and Deng, Jia},
        journal = {Advances in neural information processing systems},
        volume = {34},
        pages = {16558--16569},
        year = {2021},
    }
    ```
