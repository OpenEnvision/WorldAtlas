# Motion Smoothness

Measures temporal smoothness of the predicted video using the official interpolation-based metric path.

## Used By

- `video_dynamic`
- `experimental`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |

Reference media is not required by the official path.

## Raw Output

- Official backend: `VFIMamba`.
- Raw output has three components:
  - `mse`
  - `ssim`
  - `lpips`

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw.mse` | interpolation reconstruction error |
| `raw.ssim` | structural similarity |
| `raw.lpips` | perceptual distance |
| `normalized` | reduced benchmark score in `[0, 1]` |
| `details.native_components` | component values recorded again for convenience |

## Normalization

Component normalization:

| Component | Empirical min | Empirical max | Direction |
| --- | ---: | ---: | --- |
| `mse` | `0.0` | `82.4014` | lower is better |
| `ssim` | `0.9224` | `1.0` | higher is better |
| `lpips` | `0.0` | `0.0228` | lower is better |

Reducer:

- arithmetic mean over normalized components

## Official Rule

Official whenever it returns a numeric value in official suites. It also appears diagnostically in `experimental`.

## How To Read It

- High normalized score: transitions are temporally smooth and perceptually stable.
- Low normalized score: flicker, popping, interpolation failure, or abrupt temporal artifacts are present.

## Common Misreads

- Smoothness is not the same as motion strength.
- A nearly static video can look smooth while still failing motion-control metrics.
- Component values matter: `MSE`, `SSIM`, and `LPIPS` should be read together when debugging.

## Implementation

- `worldarena/benchmark/metrics/motion.py`
- `worldarena/benchmark/official_runtime.py`
