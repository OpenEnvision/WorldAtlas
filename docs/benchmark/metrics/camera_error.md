# Camera Error

Measures how closely the predicted video camera trajectory matches annotation-side camera poses.

## Used By

- `video_static`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |
| camera ground truth | `poses.npy` and `intrinsics.npy` from the sample annotation directory |
| scale | fixed to `1.0` in the benchmark runner |

Reference frames are not used directly. The metric operates on the predicted video plus annotation-side camera supervision.

## Raw Output

- Official backend: `DROID-SLAM`.
- The metric estimates camera motion from the prediction and compares it to the annotated camera trajectory.
- Raw output is two-component:
  - `rotation_error`
  - `translation_error`

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw.rotation_error` | camera rotation error |
| `raw.translation_error` | camera translation error |
| `normalized` | reduced benchmark score in `[0, 1]` |
| `details.frame_count` | decoded frame count |
| `details.camera_scale` | evaluation scale |

## Normalization

Component normalization:

| Component | Empirical min | Empirical max | Direction |
| --- | ---: | ---: | --- |
| rotation error | `0.0` | `15.0` | lower is better |
| translation error | `0.0` | `0.5` | lower is better |

Reducer:

- geometric mean over normalized components

## Official Rule

Official whenever pose annotations exist and the metric returns a numeric value.

If pose files are missing, the metric output stays null and does not enter aggregation.

## How To Read It

- High normalized score: the predicted camera motion stays close to the intended trajectory.
- Low normalized score: viewpoint motion is unstable, mis-scaled, or semantically wrong.

## Common Misreads

- This is camera control, not object motion control.
- A visually smooth video can still have poor camera accuracy.
- It is supervision-dependent; missing pose files mean no score, not zero score.

## Implementation

- `worldarena/benchmark/metrics/control.py`
- `worldarena/benchmark/official_runtime.py`
