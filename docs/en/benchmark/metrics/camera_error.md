# Camera Error

Camera Error quantifies **trajectory fidelity** between a camera path recovered from a generated video and annotated ground-truth extrinsics. It measures viewpoint control compliance, not object motion or instruction following.

For the direct pose-GT policy, let \(\{\mathbf{T}^{\mathrm{gt}}_t\}_{t=1}^{N}\) and \(\{\mathbf{T}^{\mathrm{pred}}_t\}_{t=1}^{N}\) denote $4 \times 4$ camera-to-world matrices from annotations and visual odometry (MegaSAM or DROID-SLAM). If the pose counts differ, the GT trajectory is uniformly resampled to the recovered pose count using linear translation interpolation and quaternion SLERP. WorldArena then estimates the full positive-scale similarity transform

\[
s^\star,\mathbf{Q}^\star,\mathbf{u}^\star
= \arg\min_{s>0,\,\mathbf{Q}\in SO(3),\,\mathbf{u}}
\sum_{t=1}^{N}
\left\|\mathbf{t}^{\mathrm{gt}}_t-
\left(s\mathbf{Q}\mathbf{t}^{\mathrm{pred}}_t+\mathbf{u}\right)\right\|_2^2
\tag{1}
\]

with Umeyama alignment. Stationary or collinear paths use corresponding camera orientations to resolve the rotational gauge before solving scale and translation. The aligned prediction is \(\tilde{\mathbf{R}}_t=\mathbf{Q}^\star\mathbf{R}^{\mathrm{pred}}_t\) and \(\tilde{\mathbf{t}}_t=s^\star\mathbf{Q}^\star\mathbf{t}^{\mathrm{pred}}_t+\mathbf{u}^\star\). Mean translation error is

\[
E_t = \frac{1}{N}\sum_{t=1}^{N} \left\|\tilde{\mathbf{t}}_t - \mathbf{t}^{\mathrm{gt}}_t\right\|_2
\tag{2}
\]

Mean rotation error (degrees) uses relative rotation \(\mathbf{R}^{\mathrm{rel}}_t = \tilde{\mathbf{R}}_t (\mathbf{R}^{\mathrm{gt}}_t)^\top\):

\[
E_r = \frac{1}{N}\sum_{t=1}^{N} \arccos\!\left(\mathrm{clip}\!\left(\frac{\mathrm{tr}(\mathbf{R}^{\mathrm{rel}}_t) - 1}{2}, -1, 1\right)\right) \cdot \frac{180}{\pi}
\tag{3}
\]

WorldArena normalizes each direct-GT component to \([0,1]\) with empirical ranges \([0, 15°]\) for rotation and \([0, 0.5]\) for translation (lower is better), then combines them by the **geometric mean**. The default `native_intended` MegaSAM policy instead evaluates recovered motion against the requested camera/action directions and returns its native direction-confidence score; it does not use the direct-GT reducer above. The metric is **official** when the selected policy has sufficient camera supervision.

Those direct-GT empirical ranges predate `sim3_v1` and must be revalidated before publishing corrected direct-GT normalized scores. This does not affect the current default `native_intended` score surface.

<!-- metric-examples:start -->
## No Verified Same-Case Pair Yet

> The retained `artifacts/reports` snapshot does not contain two distinct predictions for the same case under one comparable Camera Error protocol. The former card compared the same video through different backends, so it has been removed rather than presented as a good/bad output pair.
<!-- metric-examples:end -->

## References {#reference}

!!! quote "DROID-SLAM — Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Teed & Deng, NeurIPS 2021.
    [https://arxiv.org/abs/2108.10869](https://arxiv.org/abs/2108.10869)
