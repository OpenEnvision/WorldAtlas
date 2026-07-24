# Temporal Calibration

Temporal Calibration measures whether a generated video preserves a **stable physical clock**. When a file declares a playback frame rate, does the motion visible in the frames imply the same real-time progression? WorldArena applies this as an output-only audit in the `real_time` dimension using the wrapped Pulse-of-Motion / Visual Chronometer predictor [[Gao et al., 2026](#reference-visual-chronometer)].

This is not the same as optical flow, motion magnitude, or temporal smoothness. A rollout can look smooth frame to frame and still be temporally wrong: falls may drift as if gravity were slowed, or fast periodic motion may appear unnaturally sluggish. Visual Chronometer calls this failure mode **chronometric hallucination**—visually coherent dynamics with an ambiguous, unstable, or uncontrollable physical time scale.

The predictor estimates **Physical FPS** (`PhyFPS`) from overlapping temporal windows instead of trusting container metadata alone. WorldArena compares the mean inferred rate with the nominal playback FPS $f_{\mathrm{meta}}$ (OpenCV container FPS, or manifest fallback) and reports three official metrics: `physical_fps_error`, `physical_fps_pct_error`, and `physical_fps_intra_video_cv`. The first two capture global tempo mismatch; the third captures rhythm drift inside a single clip.

For each prediction, the runtime extracts overlapping windows with defaults `clip_length = 30`, `stride = 4`, and `resolution = 216`. Each window yields a segment estimate $\hat{f}_k$. The mean inferred physical FPS is

\[
\bar{f}_{\mathrm{phy}}
=
\frac{1}{K}\sum_{k=1}^{K} \hat{f}_k
\tag{1}
\]

Absolute and relative calibration errors are

\[
E_{\mathrm{abs}} = \left|\bar{f}_{\mathrm{phy}} - f_{\mathrm{meta}}\right|
\tag{2}
\]

\[
E_{\mathrm{pct}} = \frac{E_{\mathrm{abs}}}{f_{\mathrm{meta}}}
\tag{3}
\]

Within-video instability is the coefficient of variation over segment predictions:

\[
C_{\mathrm{intra}}
=
\frac{\operatorname{std}(\hat{f}_1,\ldots,\hat{f}_K)}{\operatorname{mean}(\hat{f}_1,\ldots,\hat{f}_K)}
\tag{4}
\]

WorldArena maps `physical_fps_error`, `physical_fps_pct_error`, and `physical_fps_intra_video_cv` to $[0,1]$ with fixed reverse scaling (`physical_fps_error`: $[0,30]$; the other two: $[0,1]$). **Higher normalized values mean better temporal calibration.** When the side runtime, checkpoint, or FPS metadata is unavailable, the metric family is reported as `not_applicable`.

## References {#reference-visual-chronometer}

!!! quote "The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"

    Xiangbo Gao, Mingyang Wu, Siyuan Yang, Jiongze Yu, Pardis Taghavi, Fangzhou Lin, and Zhengzhong Tu.  
    arXiv:2603.14375, 2026.
