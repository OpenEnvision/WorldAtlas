# Temporal Calibration

`temporal_calibration` belongs to WorldArena's top-level `real_time` dimension. It asks whether a generated video has a stable physical clock: when the file says it should play at a given FPS, does the visible motion actually imply the same passage of real time?

This is different from optical flow, motion magnitude, or smoothness. A clip can be perfectly smooth and still be temporally wrong: a falling person may drift as if gravity were slowed down, or a fast-winged insect may flap at an implausibly slow rate. The Visual Chronometer paper calls this failure mode **chronometric hallucination**: generated motion looks visually coherent but has an ambiguous, unstable, or uncontrollable physical time scale [[Gao et al., 2026](#reference-visual-chronometer)].

WorldArena computes this family with the wrapped `Pulse-of-Motion` / `Visual Chronometer` predictor. The predictor estimates **Physical Frames Per Second** (`PhyFPS`) directly from visual dynamics, rather than trusting the container's nominal metadata FPS.

## What It Measures

| Metric | Raw value | Interpretation |
| --- | --- | --- |
| `physical_fps_error` | `abs(avg_phyfps - meta_fps)` | Average physical tempo mismatch, measured in FPS |
| `physical_fps_pct_error` | `physical_fps_error / meta_fps` | Relative tempo mismatch, comparable across 12 FPS, 24 FPS, 30 FPS, and 60 FPS outputs |
| `physical_fps_intra_video_cv` | `std(segment_phyfps) / mean(segment_phyfps)` | Within-video rhythm instability across overlapping temporal segments |

Together, these metrics separate two timing failures:

- **Calibration error:** the whole video is consistently too slow or too fast relative to its declared FPS.
- **Temporal jitter:** different parts of the same video imply different physical clocks, even if the average FPS looks acceptable.

They measure physical tempo plausibility. They do **not** measure prompt following, object control, camera accuracy, perceptual quality, or whether the motion is semantically correct.

## Principle

Most video files expose a metadata FPS (`meta_fps`). That value controls playback cadence, but it does not prove that the depicted motion corresponds to real-world time. Internet videos are often slowed down, sped up, interpolated, edited, or re-encoded; large video generators trained on such data can learn visually plausible transitions without learning a stable physical clock.

Visual Chronometer formalizes the missing quantity as `PhyFPS`: the frame rate implied by the motion itself. The model predicts `log(PhyFPS)` from short frame sequences, then exponentiates the prediction to recover a physical FPS estimate. Its training data is constructed through controlled temporal resampling from high-frame-rate video, including sharp capture, motion blur, and rolling-shutter variants, so the predictor learns from motion cues instead of relying only on scene category or metadata [[Gao et al., 2026](#reference-visual-chronometer)].

WorldArena uses this idea as an output-only audit. For each generated video, it compares the physical FPS inferred from the frames with the FPS that the benchmark will use for playback. This makes temporal calibration a real-time/world-modeling check: a prediction should not merely look smooth frame to frame; it should preserve a coherent relationship between displacement and elapsed time.

## Detailed Algorithm

1. WorldArena reads the full prediction video.
2. The runtime extracts overlapping temporal windows with the current defaults:
    - `clip_length = 30`
    - `stride = 4`
    - `resolution = 216`
3. Each window is resized to `216 x 216`, normalized into `[-1, 1]`, and arranged as `[C, T, H, W]`.
4. The wrapped `Pulse-of-Motion` model predicts `log(PhyFPS)` for each window.
5. The runtime exponentiates the prediction to obtain one `predicted_phyfps` value per window.
6. It averages all window-level predictions to get `avg_phyfps`.
7. It obtains `meta_fps` from:
    - the prediction container FPS when OpenCV can read it
    - otherwise `sample.fps` from the benchmark manifest
8. It computes the three metric values:
    - `physical_fps_error = abs(avg_phyfps - meta_fps)`
    - `physical_fps_pct_error = physical_fps_error / meta_fps`
    - `physical_fps_intra_video_cv = std(segment_phyfps) / mean(segment_phyfps)`
9. WorldArena normalizes each metric with direct bounded reverse scaling:
    - `physical_fps_error`: `[0, 30]`, lower is better
    - `physical_fps_pct_error`: `[0, 1]`, lower is better
    - `physical_fps_intra_video_cv`: `[0, 1]`, lower is better

## Formula

Let the generated video be split into \(K\) overlapping windows. The predictor estimates a physical FPS \(\hat{f}_k\) for each window \(k\), and the video or manifest provides a nominal playback FPS \(f_{\mathrm{meta}}\).

!!! info "Average inferred physical FPS"

    \[
    \bar{f}_{\mathrm{phy}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    \hat{f}_k
    \]

The average calibration error is:

!!! info "Absolute FPS error"

    \[
    E_{\mathrm{abs}}
    =
    \left|
        \bar{f}_{\mathrm{phy}}
        -
        f_{\mathrm{meta}}
    \right|
    \]

The relative calibration error is:

!!! info "Relative FPS error"

    \[
    E_{\mathrm{pct}}
    =
    \frac{
        E_{\mathrm{abs}}
    }{
        f_{\mathrm{meta}}
    }
    \]

The within-video temporal instability is the coefficient of variation over the segment predictions:

!!! info "Intra-video PhyFPS variation"

    \[
    C_{\mathrm{intra}}
    =
    \frac{
        \operatorname{std}(\hat{f}_1,\ldots,\hat{f}_K)
    }{
        \operatorname{mean}(\hat{f}_1,\ldots,\hat{f}_K)
    }
    \]

WorldArena stores these three quantities as separate metrics. The normalized score for each one is a reverse-scaled value in `[0, 1]`, so higher normalized values mean better calibration.

## Reading The Scores

| Pattern | Likely diagnosis |
| --- | --- |
| low `physical_fps_error`, low `physical_fps_intra_video_cv` | well-calibrated tempo; the video has a stable physical clock |
| high `physical_fps_error`, low `physical_fps_intra_video_cv` | consistently too slow or too fast; a global retiming may improve naturalness |
| low `physical_fps_error`, high `physical_fps_intra_video_cv` | average tempo is plausible, but the clip speeds up or slows down internally |
| high `physical_fps_error`, high `physical_fps_intra_video_cv` | severe chronometric hallucination; both average tempo and local stability are unreliable |

This reading mirrors the Visual Chronometer evaluation design: the paper audits both metadata-vs-physical-FPS alignment and intra-video stability, then shows that PhyFPS-guided retiming can improve perceived naturalness [[Gao et al., 2026](#reference-visual-chronometer)].

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full prediction video |
| Reference input | none |
| Annotation input | fallback manifest FPS only if container FPS is missing |
| Raw output | one scalar per metric family member |
| Normalized output | one scalar in `[0, 1]`, where higher is better after inversion |
| Details payload | `avg_phyfps`, `segment_phyfps`, `segment_count`, `total_frames`, `meta_fps`, `meta_fps_source`, `device`, `clip_length`, `stride`, `resolution` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the side runtime, checkpoint, or FPS metadata is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | all three metrics are currently `official` when a numeric score is produced |
| Publication verdict | `Supporting real-time / temporal-calibration metric` |

- These metrics expose a failure mode that standard flow, interpolation smoothness, and perceptual quality metrics can miss.
- They depend on an external pretrained tempo estimator and on usable FPS metadata; results should be interpreted as a calibrated audit signal, not as ground truth physics.
- Re-encoding a prediction with a different container FPS can change `physical_fps_error` and `physical_fps_pct_error` even if the frame content is identical.
- `physical_fps_intra_video_cv` is metadata-independent after segment prediction, making it especially useful for detecting local rhythm drift.
- Recommended use: report these metrics next to motion smoothness, motion magnitude, and semantic/control metrics. A strong world-model output should be smooth, semantically correct, and physically timed.

## References {#reference-visual-chronometer}

!!! quote "The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"

    Xiangbo Gao, Mingyang Wu, Siyuan Yang, Jiongze Yu, Pardis Taghavi, Fangzhou Lin, and Zhengzhong Tu.  
    arXiv:2603.14375, 2026.

??? note "BibTeX"

    ```bibtex
    @article{gao2026pulse,
        title = {The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics},
        author = {Gao, Xiangbo and Wu, Mingyang and Yang, Siyuan and Yu, Jiongze and Taghavi, Pardis and Lin, Fangzhou and Tu, Zhengzhong},
        journal = {arXiv preprint arXiv:2603.14375},
        year = {2026},
    }
    ```
