# Temporal Calibration

The temporal calibration dimension asks a different question from optical-flow or smoothness metrics: does the generated video *move at the physical rate implied by its FPS*?

WorldArena currently computes this dimension with `Pulse-of-Motion`, routed through the unified metric registry instead of a separate evaluation pipeline.

## Metrics

- `physical_fps_error`: absolute difference between estimated physical FPS and the video's metadata FPS
- `physical_fps_pct_error`: relative FPS error, normalized by metadata FPS
- `physical_fps_intra_video_cv`: coefficient of variation across per-segment physical FPS estimates

Lower is better for all three metrics.

## Inputs

The backend consumes the prediction video only. It reads:

- the predicted video frames
- the container FPS when available
- the sample FPS from the manifest as a fallback when container metadata is missing

## Backend Contract

By default, WorldArena looks for the source checkout at `third_party/Pulse-of-Motion`. You can override that with:

```bash
export WORLDARENA_PULSE_OF_MOTION_ROOT=/path/to/Pulse-of-Motion
```

The runtime expects:

- `inference/predict.py`
- `inference/configs/config_fps.yaml`

The upstream checkpoint `inference/ckpts/vc_common_10_60fps.ckpt` can either be present locally or downloaded on first use through `huggingface_hub`.

Shared runtime knobs can be set once under `benchmark.metric_runtime.temporal_calibration` in `config/benchmark.yaml`. WorldArena forwards `device`, `clip_length`, `stride`, and `resolution` to the upstream predictor.

Check readiness with:

```bash
uv run worldarena check-env
```

## Reporting

These metrics are grouped under the `temporal_calibration` dimension and can be assigned to the same `video_dynamic` suite as motion-control metrics. They should not be interpreted as substitutes for `motion_accuracy`, `motion_magnitude`, or `motion_smoothness`; they measure temporal plausibility, not control fidelity or rendering stability.
