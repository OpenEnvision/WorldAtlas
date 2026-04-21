# Prompt Alignment

Measures whether the prediction matches the benchmark target instruction at the sampled anchor frames.

## Used By

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

## Inputs

| Input | Source |
| --- | --- |
| prediction anchor frames | sampled from the prediction under the official protocol |
| target instruction | `sample.prompt_target` |
| supervision gate | `sample.has_instruction` and `sample.conditioning_strategy != "self"` |

Reference media is not used by the official score path.

## Raw Output

- Official backend: `CLIPScore` reference stack.
- Each anchor frame is scored against `prompt_target`.
- Native `CLIPScore` values are stored in `details.native_frame_scores`.
- The metric converts the native `0..100` score range into `0..1` frame scores and averages them.

Current payload shape:

| Field | Meaning |
| --- | --- |
| `raw` | mean normalized frame score |
| `normalized` | benchmark-scaled score in `[0, 1]` |
| `details.frame_scores` | per-anchor normalized scores |
| `details.native_frame_scores` | per-anchor native `CLIPScore` values |
| `details.frame_indices` | anchor indices used for evaluation |

## Normalization

Current config:

- `lower = 0.0`
- `upper = 1.0`
- `higher_is_better = true`

That makes the current benchmark normalization effectively identity on the reference output.

## Official Rule

- `official` only when the sample has explicit instruction supervision and is not self-conditioned.
- `weak_supervision` otherwise.

The key case is image benchmarking: current image samples are self-conditioned, so `prompt_alignment` is computed there only as a diagnostic.

## How To Read It

- High: the requested scene, object, or motion semantics are visible at anchor frames.
- Low: the generated content drifts from the target instruction.

This metric does not measure realism, temporal smoothness, or geometric consistency.

## Common Misreads

- A visually impressive result can still score low if it misses the requested semantics.
- Generic prompts are easier than compositional prompts.
- Strong style matching does not guarantee high prompt alignment.

## Implementation

- `worldarena/benchmark/metrics/alignment.py`
- `worldarena/benchmark/reference_runtime.py`
