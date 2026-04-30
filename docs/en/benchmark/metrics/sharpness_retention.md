# Sharpness Retention

`sharpness_retention` measures whether a generated video preserves visible texture sharpness after the initial frames.

## What It Measures

- blur accumulation over time
- loss of edge energy after the start of a rollout
- high-frequency artifact filtering through a BRISQUE gate when that backend is enabled

It is inspired by the iWorld-Bench sharpness-retention slot, whose public description mentions Tenengrad and a BRISQUE-triggered circuit breaker. Because the official implementation is not public, this metric is a WorldArena-local diagnostic.

## Detailed Algorithm

1. WorldArena reads the prediction's sampled video frames.
2. It computes a Tenengrad sharpness score for each frame using Sobel gradient energy.
3. The baseline is the median Tenengrad score over the first configured baseline frames.
4. Later frames are scored with `retention = clamp(tenengrad_t / baseline, 0, 1)`.
5. The base sharpness-retention score is the mean later-frame retention.
6. With the default `tenengrad_brisque` backend, BRISQUE is also evaluated on the same frames.
7. Mean BRISQUE above the configured bad threshold gates the score toward zero; missing BRISQUE dependencies are reported as explicit metric errors, not silently ignored.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | none |
| Annotation input | none |
| Raw output | scalar in `[0, 1]`; higher means better retained sharpness |
| Normalized output | identity under the default config |
| Details payload | Tenengrad scores, baseline, retention scores, BRISQUE scores, BRISQUE gate |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if fewer than two frames are available, the baseline texture is too low, or the configured BRISQUE backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: it catches blur decay that may not strongly affect semantic alignment.
- Why it is limited: high edge energy can come from artifacts, which is why the default backend gates Tenengrad with BRISQUE.
- Recommended use: report as a diagnostic quality metric alongside `perceptual_quality` rather than as a main score.
