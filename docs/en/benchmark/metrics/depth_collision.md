# Depth Collision

Depth Collision is a binary flag for implausibly near geometry directly ahead of the camera. Video Depth Anything supplies per-frame metric depth; a center crop tests whether the forward viewing corridor becomes abnormally close during rollout.

For frame \(t\) with depth map \(D_t\) and center crop \(\mathcal{C}\),

\[
\bar{d}_t = \frac{1}{|\mathcal{C}|}\sum_{(x,y)\in\mathcal{C}} D_t(x,y)
\tag{1}
\]

Given threshold \(\tau\) (default \(1.0\)), the collision indicator is

\[
e_{\mathrm{collision}} = \mathbf{1}\!\left[\min_t \bar{d}_t < \tau\right]
\tag{2}
\]

Raw score \(e_{\mathrm{collision}} \in \{0,1\}\) is reverse-normalized so that no collision maps to higher score. The flag proxies near-depth failure (wall penetration, face-close motion) under estimated depth—it is not a physics-engine contact test.

<!-- metric-examples:start -->
## No Verified Same-Case Pair Yet

> The retained formal reports currently provide only one prediction run for this metric, so no verified same-case high/low output pair is available. Different-case extrema are intentionally not shown.
<!-- metric-examples:end -->

## References {#reference}

!!! quote "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"

    Chen et al., arXiv:2501.12375, 2025.
