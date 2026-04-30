# Long-Sequence Diagnostics

WorldArena bundles a native `Long-Sequence` backend to extract long-horizon motion and quality diagnostics from full prediction videos. These metrics do not only inspect single frames or adjacent frame pairs. They split a generated video into short clips, then analyze longer-range motion continuity and visual quality.

The main question is whether a generated video still preserves motion continuity and image quality over a longer horizon.

These metrics are prediction-only diagnostics. They do not use the benchmark reference rollout or annotations. They are best used to explain long-video degradation modes such as motion collapse, abrupt temporal artifacts, or quality decay over time.

## Covered Metrics

| Metric name | Wrapped Long-Sequence dimension | Role in WorldArena |
| --- | --- | --- |
| `long_sequence_motion_smoothness` | `motion_smoothness` | motion smoothness based on video-frame-interpolation error |
| `long_sequence_dynamic_degree` | `dynamic_degree` | optical-flow-based probe for whether enough motion occurs |
| `long_sequence_aesthetic_quality` | `aesthetic_quality` | no-reference aesthetic quality |
| `long_sequence_imaging_quality` | `imaging_quality` | MUSIQ-based image clarity and degradation quality |

## Principle

Short-range metrics can miss failures that appear only after a longer rollout. The Long-Sequence backend first splits the full video into short clips, then applies native long-video routines to motion and quality dimensions.

`motion_smoothness` uses AMT-S to interpolate middle frames and compares the interpolation with real middle frames. `dynamic_degree` uses RAFT flow strength to detect whether motion is sufficiently present. `aesthetic_quality` and `imaging_quality` use no-reference quality models over clip frames.

## Detailed Algorithm

1. WorldArena reads the full prediction video and prepares a Long-Sequence input workspace.
2. `worldarena.benchmark.long_video` splits the video into `split_clip/` inputs and builds the minimal manifest required by each dimension.
3. A dispatcher routes each requested dimension to the corresponding implementation:
    - `motion_smoothness`: AMT-S interpolation, comparing predicted middle frames with real middle frames.
    - `dynamic_degree`: RAFT optical flow, counting whether high-motion transitions exceed the threshold.
    - `aesthetic_quality`: no-reference aesthetic scorer over clip frames.
    - `imaging_quality`: MUSIQ over clip frames.
4. Each WorldArena metric reads only its own wrapped dimension.
5. The wrapper records runtime metadata, including device, source root, clip count, per-video scores, and dimension-specific details.
6. Each metric is independently normalized according to the `[0, 1]` range in `config/benchmark.yaml`; all four metrics are higher-is-better.

## Key Formulas

For `motion_smoothness`, AMT-S interpolates middle frames from evenly sampled frames and compares interpolated and real middle frames using mean absolute difference:

!!! info "Motion smoothness"

    \[
    s_{\mathrm{smooth}}
    =
    \operatorname{clamp}
    \left(
        \frac{255-\operatorname{mean}(|I^{real}_{mid}-I^{interp}_{mid}|)}{255},
        0,
        1
    \right)
    \]

For `dynamic_degree`, RAFT estimates optical flow between adjacent sampled frames and uses the mean magnitude of the top 5% highest-motion pixels as the transition motion score. If enough transitions exceed the motion threshold, the clip is marked dynamic and outputs `1.0`; otherwise it outputs `0.0`.

For `aesthetic_quality` and `imaging_quality`, the backend scores frames in each clip with no-reference quality models, then averages over clips, videos, and the sample batch.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | one `[0, 1]` scalar per wrapped Long-Sequence metric; higher is better |
| Normalized output | one `[0, 1]` scalar per metric; same direction as raw under the current config |
| Details payload | `dimension`, `duration_seconds`, `source_root`, `runtime_device`, `clip_result_count`, `video_count`, `per_video_scores`, frame index metadata, plus dimension-specific fields |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string when the bundled backend or local runtime dependencies are missing, or when `Long-Sequence` does not return a value for the requested dimension |

## Runtime Notes

- The current wrapper reads the side runtime from `benchmark.metric_runtime.long_sequence`.
- AMT / RAFT source is fixed at `thirdparty/AMT` and `thirdparty/RAFT`.
- Long-sequence checkpoints resolve from `benchmark.metric_runtime.long_sequence` into the single checkpoint root:
  `ckpt/amt_model/amt-s.pth`, `ckpt/raft_model/models/raft-things.pth`, and `ckpt/musiq/musiq_spaq_ckpt-358bb6af.pth`.
- Shared HF/Torch caches are pinned to `ckpt/huggingface`, `ckpt/transformers`, and `ckpt/torch`.
- The native path no longer depends on `decord`, `moviepy`, or `easydict`. Quality paths still depend on `pyiqa` plus a compatible `timm` install.
- `use_semantic_splitting=true` additionally requires `scenedetect` in the runtime environment.
- Common runtime knobs include `device`, `use_semantic_splitting`, `clip_duration_seconds`, `clip_fps`, `clip_length_config`, and `imaging_quality_preprocessing_mode`.

## Integration Notes

- Current aggregation: `diagnostic` only, emitted for `image_static`, `image_dynamic`, and `video_dynamic`.
- These metrics are useful for long-horizon analysis, but they intentionally stay out of the official headline vector.
- Interpret them together with `style_consistency`, `perceptual_quality`, `motion_magnitude`, and `motion_smoothness`: high Long-Sequence scores indicate better long-horizon motion and quality statistics, not perfect prompt following or physically correct motion.
