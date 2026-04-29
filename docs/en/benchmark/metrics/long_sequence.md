# Long-Sequence Diagnostics

WorldArena bundles a native `Long-Sequence` backend to emit long-video diagnostic metrics for `image_static`, `image_dynamic`, and `video_dynamic` video predictions.

## Covered Metrics

| Metric name | Wrapped Long-Sequence dimension | Role in WorldArena |
| --- | --- | --- |
| `long_sequence_subject_consistency` | `subject_consistency` | diagnostic long-horizon identity stability |
| `long_sequence_background_consistency` | `background_consistency` | diagnostic long-horizon background stability |
| `long_sequence_motion_smoothness` | `motion_smoothness` | diagnostic long-horizon temporal smoothness |
| `long_sequence_dynamic_degree` | `dynamic_degree` | diagnostic motion-strength probe |
| `long_sequence_aesthetic_quality` | `aesthetic_quality` | diagnostic visual-quality probe |
| `long_sequence_imaging_quality` | `imaging_quality` | diagnostic image-quality probe |

## Detailed Algorithm

1. WorldArena resolves its native long-sequence assets from two places: built-in mapping / slow-fast defaults live in `worldarena/benchmark/long_sequence_params.py`, while the AMT / RAFT model code is discovered from official upstream checkouts under `thirdparty/AMT` and `thirdparty/RAFT`.
2. The long-video orchestration is handled natively by WorldArena: `worldarena.benchmark.long_video` prepares `split_clip/` inputs and builds the minimal manifest consumed by the algorithm implementations.
3. A fixed WorldArena dispatcher then routes each requested dimension to its corresponding implementation.
4. Each metric reads the score that corresponds to its own wrapped dimension.
5. The current wrapper records extra runtime metadata such as the effective device, source root, and dimension-specific details returned by the bundled runtime.
6. Each metric is then normalized independently with the benchmark ranges configured in `config/benchmark.yaml`.

All six currently shipped long-sequence dimensions are now routed through WorldArena-native long-video logic. Shared feature, quality, and flow backends own the model loading and per-clip scoring.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | full prediction video |
| Reference input | none |
| Annotation input | none |
| Raw output | one scalar per wrapped Long-Sequence metric |
| Normalized output | one scalar in `[0, 1]` per metric |
| Details payload | `dimension`, `duration_seconds`, `source_root`, `runtime_device`, frame index metadata, plus dimension-specific fields returned by `Long-Sequence` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string when the bundled backend or runtime dependencies are missing, or when `Long-Sequence` does not return a value for the requested dimension |

## Runtime Notes

- The current wrapper reads the side runtime from `benchmark.metric_runtime.long_sequence`.
- Native mapping / slow-fast defaults live in `worldarena/benchmark/long_sequence_params.py`.
- AMT / RAFT source is fixed at `thirdparty/AMT` and `thirdparty/RAFT`.
- Long-sequence checkpoints now resolve from `benchmark.metric_runtime.long_sequence` into the single checkpoint root:
  `ckpt/amt_model/amt-s.pth`, `ckpt/raft_model/models/raft-things.pth`, and `ckpt/musiq/musiq_spaq_ckpt-358bb6af.pth`.
- Shared HF/Torch caches are pinned to `ckpt/huggingface`, `ckpt/transformers`, and `ckpt/torch`.
- The native path no longer depends on `decord`, `moviepy`, or `easydict`. Quality-related paths still depend on `pyiqa` plus a compatible `timm` install.
- `use_semantic_splitting=true` additionally requires `scenedetect` in the runtime environment.
- Common runtime knobs include `device`, `use_semantic_splitting`, `bg_clip2clip_feat_extractor`, `sb_clip2clip_feat_extractor`, and `clip_length_config`.
- These metrics are prediction-only diagnostics. They do not use the benchmark reference rollout.

## Integration Notes

- Current aggregation: `diagnostic` only, emitted for `image_static`, `image_dynamic`, and `video_dynamic`.
- These metrics are useful for long-horizon analysis, but they intentionally stay out of the official headline vector.
