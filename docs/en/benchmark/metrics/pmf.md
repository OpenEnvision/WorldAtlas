# Physical Motion Fidelity

`pmf` measures whether the prediction preserves the same broad motion dynamics as the reference rollout, while deliberately reducing sensitivity to absolute spatial position and global brightness.

WorldArena currently implements PMF directly in-repo with a NumPy FFT backend. The metric follows the Fourier-based motion-fidelity idea used by PhysInOne [[Zhou et al., 2026](#reference)]: instead of comparing prediction and reference videos pixel by pixel, it compares their normalized motion-energy distributions in the joint space-time frequency domain.

## What It Measures

- similarity of spatiotemporal motion-energy patterns between prediction and reference
- robustness to simple spatial/temporal shifts and global brightness scaling
- whether the generated rollout keeps the broad dynamic rhythm and frequency structure of the reference video
- not prompt following, not object identity, and not pixel-perfect reconstruction

## Principle

The core motivation for a Fourier-based metric is that, when evaluating generated videos, we often care more about whether the motion dynamics are similar than whether every pixel is perfectly aligned.

If a video is shifted in space, delayed in time, or globally rescaled in brightness, its physical motion pattern may still match the reference. Pixelwise metrics punish these differences strongly. PMF tries to ignore this kind of initial-state, phase, and brightness mismatch, and instead compares the video's spatiotemporal spectral energy structure.

Concretely, PMF treats a video as an \(H\times W\times T\) spatiotemporal signal and applies a 3D DFT to each RGB channel. The Fourier magnitude describes how much energy exists at each spatial and temporal frequency, while the phase encodes translation-like information such as position and temporal offset. In the normalized energy spectrum, global brightness scaling cancels in the denominator, and simple spatiotemporal shifts change phase without changing magnitude energy. Comparing normalized energy spectra therefore focuses the score on motion dynamics rather than start position, absolute brightness, or pixel alignment.

## Detailed Algorithm

1. WorldArena reads the rollout evaluation window from the prediction and reference videos.
2. If the two videos expose different numbers of sampled frames, it uniformly resamples both to the shorter length.
3. If the resolutions differ, it resizes prediction frames to the reference resolution.
4. It stacks each video as an RGB tensor with shape `[T, H, W, C]`.
5. It applies a 3D discrete Fourier transform over `H x W x T` independently for each RGB channel.
6. It converts the Fourier coefficients into a summed power spectrum:
    - `power(u, v, s) = sum_c |F_c(u, v, s)|^2`
7. It normalizes that power spectrum into a probability-like energy distribution:
    - `E(u, v, s) = power(u, v, s) / sum power`
8. It computes total variation distance between prediction and reference energy distributions:
    - `tv = 0.5 * sum |E_pred - E_ref|`
9. It reports:
    - `pmf = -log(max(tv, 1e-12))`
10. WorldArena then applies bounded normalization with:
    - `lower = 0.0`
    - `upper = 6.0`
    - higher is better

Higher PMF means the two videos have more similar motion-energy structure. Perfect spectral agreement saturates at the epsilon floor rather than returning mathematical infinity.

## Spectrum Definition

Given a video \(V\), let \(c\in\{0,1,2\}\) be the RGB channel and let \((h,w,t)\) be the spatial and temporal coordinates. For each channel, PMF computes the 3D discrete Fourier transform:

!!! info "3D DFT"

    \[
    \tilde{V}(c;u,v,s)
    =
    \sum_{h=0}^{H-1}
    \sum_{w=0}^{W-1}
    \sum_{t=0}^{T-1}
    V(c;h,w,t)
    \exp
    \left(
        -2\pi i
        \left(
            \frac{uh}{H}
            +
            \frac{vw}{W}
            +
            \frac{st}{T}
        \right)
    \right)
    \]

Here \((u,v)\) are spatial-frequency indices and \(s\) is the temporal-frequency index. Magnitude and phase are:

!!! info "Magnitude and phase"

    \[
    A_{c;u,v,s}
    =
    \left|
        \tilde{V}(c;u,v,s)
    \right|,
    \qquad
    \psi_{c;u,v,s}
    =
    \arg
    \left(
        \tilde{V}(c;u,v,s)
    \right)
    \]

PMF uses the RGB-combined normalized energy spectrum:

!!! info "Normalized energy spectrum"

    \[
    E_{u,v,s}
    =
    \frac{
        \sum_{c=0}^{2}
        \left|
            \tilde{V}(c;u,v,s)
        \right|^2
    }{
        \sum_{u',v',s'}
        \sum_{c=0}^{2}
        \left|
            \tilde{V}(c;u',v',s')
        \right|^2
    }
    \]

This makes \(E\) behave like a distribution over where the video's spatiotemporal motion energy is concentrated.

## Invariance Intuition

Consider a video after a spatiotemporal shift and brightness scaling:

!!! info "Shift and brightness scaling"

    \[
    V'(c;h,w,t)
    =
    \lambda
    V(c;h+\delta_h,w+\delta_w,t+\delta_t),
    \qquad
    \lambda\in(0,1]
    \]

Under boundary assumptions such as duplicate padding, the Fourier transform changes as:

!!! info "Frequency-domain change"

    \[
    \tilde{V}'(c;u,v,s)
    =
    \lambda
    e^{i\delta\psi}
    \tilde{V}(c;u,v,s)
    \]

with phase offset:

!!! info "Phase offset"

    \[
    \delta\psi
    =
    2\pi
    \left(
        \frac{u\delta_h}{H}
        +
        \frac{v\delta_w}{W}
        +
        \frac{s\delta_t}{T}
    \right)
    \]

Thus a spatiotemporal shift changes phase, while global brightness scaling multiplies magnitudes by a common factor. In the normalized energy spectrum \(E_{u,v,s}\), the \(\lambda^2\) factor cancels between numerator and denominator, and the phase term does not affect squared magnitude. This is why PMF is more robust to simple shifts and brightness scaling than pixelwise comparisons.

## PMF Formula

Given a generated video \(V_{\mathrm{gen}}\) and reference video \(V_{\mathrm{ref}}\), compute normalized energy spectra \(E_{\mathrm{gen}}\) and \(E_{\mathrm{ref}}\). Their total variation distance is:

!!! info "Total variation"

    \[
    d_{\mathrm{TV}}
    \left(
        E_{\mathrm{gen}},
        E_{\mathrm{ref}}
    \right)
    =
    \frac{1}{2}
    \sum_{u,v,s}
    \left|
        E^{\mathrm{gen}}_{u,v,s}
        -
        E^{\mathrm{ref}}_{u,v,s}
    \right|
    \]

PMF is:

!!! info "Physical Motion Fidelity"

    \[
    \operatorname{PMF}
    \left(
        V_{\mathrm{gen}},
        V_{\mathrm{ref}}
    \right)
    =
    -\ln
    \left(
        \max
        \left(
            d_{\mathrm{TV}}
            \left(
                E_{\mathrm{gen}},
                E_{\mathrm{ref}}
            \right),
            10^{-12}
        \right)
    \right)
    \]

Smaller TV distance means the two normalized spatiotemporal spectra are more similar, so PMF becomes higher. The current implementation uses `epsilon = 1e-12` to avoid \(-\ln(0)\) when spectra are identical.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | rollout-window prediction video |
| Reference input | rollout-window reference video |
| Annotation input | none |
| Raw output | one scalar PMF score |
| Normalized output | one scalar in `[0, 1]` after direct bounded scaling |
| Details payload | `tv_distance`, `epsilon_floor`, `aligned_frame_count`, `reference_resolution`, `prediction_frame_count`, `reference_frame_count`, `temporal_alignment`, `prediction_resized`, `reference_resized`, plus frame-index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if either video has fewer than two frames or decoding/alignment fails |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `official` for `image_static`, `image_dynamic`, and `video_dynamic` when a numeric score is produced |
| Publication verdict | `Main-table usable, with the caveat that it is a reference-based Fourier dynamics metric` |

- This metric is substantially more informative than plain motion-amplitude probes when the question is whether the generated rollout follows the same broad physical dynamics as the reference.
- Pure image references usually produce no score because PMF needs at least two reference frames.
- It is still not a complete correctness metric. Two videos can share similar motion spectra while differing in object identity, causal interactions, or semantic correctness.
- Because it is reference-paired, report it with control, persistence, and quality metrics rather than using it as a single-number claim about world modeling.

## Reference {#reference}

!!! quote "PhysInOne: Visual Physics Learning and Reasoning in One Suite"

    Siyuan Zhou, Hejun Wang, Hu Cheng, Jinxi Li, Dongsheng Wang, Junwei Jiang, Yixiao Jin, Jiayue Huang, Shiwei Mao, Shangjia Liu, Yafei Yang, Hongkang Song, Shenxing Wei, Zihui Zhang, Peng Huang, Shijie Liu, Zhengli Hao, Hao Li, Yitian Li, Wenqi Zhou, Zhihan Zhao, Zongqi He, Hongtao Wen, Shouwang Huang, Peng Yun, Bowen Cheng, Pok Kazaf Fu, Wai Kit Lai, Jiahao Chen, Kaiyuan Wang, Zhixuan Sun, Ziqi Li, Haochen Hu, Di Zhang, Chun Ho Yuen, Bing Wang, Zhihua Wang, Chuhang Zou, and Bo Yang.<br>
    arXiv:2604.09415, 2026.<br>
    [https://arxiv.org/abs/2604.09415](https://arxiv.org/abs/2604.09415)

??? note "BibTeX"

    ```bibtex
    @misc{zhou2026physinonevisualphysicslearning,
        title = {PhysInOne: Visual Physics Learning and Reasoning in One Suite},
        author = {Siyuan Zhou and Hejun Wang and Hu Cheng and Jinxi Li and Dongsheng Wang and Junwei Jiang and Yixiao Jin and Jiayue Huang and Shiwei Mao and Shangjia Liu and Yafei Yang and Hongkang Song and Shenxing Wei and Zihui Zhang and Peng Huang and Shijie Liu and Zhengli Hao and Hao Li and Yitian Li and Wenqi Zhou and Zhihan Zhao and Zongqi He and Hongtao Wen and Shouwang Huang and Peng Yun and Bowen Cheng and Pok Kazaf Fu and Wai Kit Lai and Jiahao Chen and Kaiyuan Wang and Zhixuan Sun and Ziqi Li and Haochen Hu and Di Zhang and Chun Ho Yuen and Bing Wang and Zhihua Wang and Chuhang Zou and Bo Yang},
        year = {2026},
        eprint = {2604.09415},
        archivePrefix = {arXiv},
        primaryClass = {cs.CV},
        url = {https://arxiv.org/abs/2604.09415},
    }
    ```
