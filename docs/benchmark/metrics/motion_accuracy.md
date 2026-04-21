# Motion Accuracy

Measures whether motion inside benchmark-supervised dynamic regions matches the intended motion pattern.

## Used By

- `video_dynamic`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |
| mask directory | sample-level segmentation masks |
| generation type | `i2v` for first-frame conditioning, otherwise `t2v` |

The benchmark runner chooses `i2v` when `conditioning_strategy == "first_frame"`.

## Raw Output

- Official backend: `SAM2 + SEA-RAFT`.
- Raw output is one scalar motion-control score from the official runtime.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | native motion-accuracy score |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.mask_count` | number of masks used |
| `details.generate_type` | `i2v` or `t2v` |
| `details.frame_count` | decoded frame count |

## Normalization

Current z-score config:

| Parameter | Value |
| --- | ---: |
| `avg` | `-0.1965` |
| `std` | `2.3687` |
| `z_min` | `-1.6147` |
| `z_max` | `1.5180` |
| output range | `[0.25, 0.75]` |
| direction | higher is better |

## Official Rule

Official only when segmentation masks exist and the metric returns a numeric value.

## How To Read It

- High: motion inside supervised dynamic regions follows the intended behavior more closely.
- Low: the right region may exist, but its motion is too weak, wrong, or temporally inconsistent.

## Common Misreads

- This is not a general motion-strength score.
- It is supervision-dependent; missing masks mean no official score.
- Strong `motion_magnitude` does not guarantee strong `motion_accuracy`.

## Implementation

- `worldarena/benchmark/metrics/control.py`
- `worldarena/benchmark/official_runtime.py`

## Runtime Notes

- Default benchmark config enables `sam2_offload_video_to_cpu` and `sam2_offload_state_to_cpu`.
- Multi-GPU hosts can also split devices through `benchmark.metric_runtime.motion_accuracy`, for example `sam2_device: cuda:0` and `flow_device: cuda:1`.
