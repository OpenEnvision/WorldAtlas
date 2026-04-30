# Motion Smoothness

`motion_smoothness` measures whether the generated motion is smooth, continuous, and compatible with the motion prior learned by a video frame interpolation model.

The core idea is simple: if a video's motion is smooth, a standard frame-interpolation model should be able to reconstruct the discarded odd frames from the neighboring even frames. The closer the reconstructed frames are to the original odd frames, the more temporally interpolatable the video is, and the fewer abrupt transitions or flicker artifacts it likely contains.

## What It Measures

- temporal stability and interpolatability
- abrupt transitions, flicker, and local structural breaks between adjacent frames
- whether generated motion follows common video-frame-interpolation motion priors
- not motion strength and not semantic correctness

## Principle

Video frame interpolation models learn the transition problem: given two surrounding frames, what should the middle frame look like? That makes them useful as a motion prior. If a generated video has smooth trajectories, the even frames should contain enough information to infer the missing odd frames. If the video has jumps, jitter, flicker, or identity drift, the interpolation model will struggle to reconstruct the original odd frames.

WorldArena currently uses the wrapped `VFIMamba` [[Zhang et al., 2024](#reference-vfimamba)] backend as the interpolation model. The metric does not need a reference video or annotations; it only checks temporal continuity inside the prediction video. It drops odd frames, reconstructs them from neighboring even frames, then compares reconstructed and original odd frames with MSE, SSIM [[Wang et al., 2004](#reference-ssim)], and LPIPS [[Zhang et al., 2018](#reference-lpips)].

The three components cover different error views:

- `MSE` measures pixel-level reconstruction error; lower is better.
- `SSIM` measures local structural similarity; higher is better.
- `LPIPS` measures perceptual feature distance; lower is better.

After component normalization, WorldArena averages the three components to combine pixel error, structural preservation, and perceptual consistency.

## Detailed Algorithm

1. WorldArena decodes the full prediction video into frame PNGs.
2. The wrapped third-party metric splits the decoded frames into:
    - even frames: `I_0, I_2, I_4, ...`
    - odd frames: `I_1, I_3, I_5, ...`
3. After odd frames are removed, the even frames form a lower-frame-rate video.
4. For each pair of neighboring even frames `(I_{2k}, I_{2k+2})`, `VFIMamba` [[Zhang et al., 2024](#reference-vfimamba)] predicts the middle frame `Î_{2k+1}`.
5. The predicted middle frame is compared with the actual odd frame `I_{2k+1}` using three component metrics:
    - `MSE`
    - `SSIM`
    - `LPIPS`
6. The raw metric is a three-component average across all interpolation windows:
    - `raw.mse = mean window MSE`
    - `raw.ssim = mean window SSIM`
    - `raw.lpips = mean window LPIPS`
7. WorldArena normalizes each component independently:
    - `mse` in `[0.0, 82.4014]`, lower is better
    - `ssim` in `[0.9224, 1.0]`, higher is better
    - `lpips` in `[0.0, 0.0228]`, lower is better
8. The final normalized score is the arithmetic mean of the three normalized components.

## Error Formula

Given generated video frames \(\{I_0,I_1,I_2,\ldots\}\), the interpolation model \(\Phi\) predicts the middle frame for the \(k\)-th window from adjacent even frames:

!!! info "Interpolation reconstruction"

    \[
    \hat{I}_{2k+1}
    =
    \Phi(I_{2k}, I_{2k+2})
    \]

The prediction \(\hat{I}_{2k+1}\) is compared with the original discarded odd frame \(I_{2k+1}\). Pixel-level error is:

!!! info "MSE component"

    \[
    e^{\mathrm{mse}}_k
    =
    \operatorname{MSE}
    \left(
        \hat{I}_{2k+1},
        I_{2k+1}
    \right)
    \]

The structural similarity component is:

!!! info "SSIM component"

    \[
    s^{\mathrm{ssim}}_k
    =
    \operatorname{SSIM}
    \left(
        \hat{I}_{2k+1},
        I_{2k+1}
    \right)
    \]

The perceptual distance component is:

!!! info "LPIPS component"

    \[
    e^{\mathrm{lpips}}_k
    =
    \operatorname{LPIPS}
    \left(
        \hat{I}_{2k+1},
        I_{2k+1}
    \right)
    \]

Averaging over all interpolation windows gives the raw three-component output:

!!! info "Raw components"

    \[
    E_{\mathrm{mse}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    e^{\mathrm{mse}}_k,
    \qquad
    S_{\mathrm{ssim}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    s^{\mathrm{ssim}}_k,
    \qquad
    E_{\mathrm{lpips}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    e^{\mathrm{lpips}}_k
    \]

WorldArena maps all three components to `[0, 1]`: MSE and LPIPS use reverse scaling, while SSIM uses forward scaling. The final score is:

!!! info "Final score"

    \[
    s_{\mathrm{smooth}}
    =
    \frac{
        \tilde{s}_{\mathrm{mse}}
        +
        \tilde{s}_{\mathrm{ssim}}
        +
        \tilde{s}_{\mathrm{lpips}}
    }{3}
    \]

Higher scores mean the interpolation model can more easily reconstruct discarded odd frames from neighboring even frames, which indicates smoother generated motion.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full decoded prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | `{"mse": float, "ssim": float, "lpips": float}` |
| Normalized output | one scalar in `[0, 1]` after component normalization and arithmetic mean reduction |
| Details payload | `frame_count`, `native_components`, `frame_indices`, `sampled_frame_indices` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the interpolation runtime fails |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` when a numeric score is produced |
| Publication verdict | `Supporting temporal-quality metric; interpret with motion amount and semantic/control metrics` |

- This is a legitimate temporal-stability signal, but it does not tell you whether the motion is semantically correct.
- A nearly static video can score well here while still being a bad generative result.
- Recommended use: report it next to motion magnitude or semantic metrics that test different failure modes.

## References {#reference-vfimamba}

!!! quote "VFIMamba: Video Frame Interpolation with State Space Models"

    Guozhen Zhang, Chunxu Liu, Yutao Cui, Xiaotong Zhao, Kai Ma, and Limin Wang.<br>
    Advances in Neural Information Processing Systems, 37:107225-107248, 2024.

??? note "BibTeX"

    ```bibtex
    @article{zhang2024vfimamba,
        title = {Vfimamba: Video frame interpolation with state space models},
        author = {Zhang, Guozhen and Liu, Chunxu and Cui, Yutao and Zhao, Xiaotong and Ma, Kai and Wang, Limin},
        journal = {Advances in Neural Information Processing Systems},
        volume = {37},
        pages = {107225--107248},
        year = {2024},
    }
    ```

### SSIM {#reference-ssim}

!!! quote "Image Quality Assessment: From Error Visibility to Structural Similarity"

    Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli.<br>
    IEEE Transactions on Image Processing, 13(4):600-612, 2004.

??? note "BibTeX"

    ```bibtex
    @article{wang2004image,
        title = {Image quality assessment: from error visibility to structural similarity},
        author = {Wang, Zhou and Bovik, Alan C and Sheikh, Hamid R and Simoncelli, Eero P},
        journal = {IEEE transactions on image processing},
        volume = {13},
        number = {4},
        pages = {600--612},
        year = {2004},
        publisher = {IEEE},
    }
    ```

### LPIPS {#reference-lpips}

!!! quote "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.<br>
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

??? note "BibTeX"

    ```bibtex
    @inproceedings{zhang2018unreasonable,
        title = {The unreasonable effectiveness of deep features as a perceptual metric},
        author = {Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages = {586--595},
        year = {2018},
    }
    ```
