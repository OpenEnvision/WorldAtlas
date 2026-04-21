# Object Detection

Measures whether required scene objects can be recovered from the predicted video under the benchmark prompt-matching contract.

## Used By

- `video_static`

## Inputs

| Input | Source |
| --- | --- |
| predicted video | evaluated rollout |
| object prompt candidates | derived from the annotation directory |
| prompt source | explicit object fields or caption-derived fallback |

The metric builds a prompt string from available object phrases and passes it to the official detector.

## Raw Output

- Official backend: `Grounding DINO`.
- The benchmark runner stores:
  - `details.prompt_source`
  - `details.prompt_candidates`
  - `details.prompt_string`
- Raw output is one scalar detection score from the official prompt-matching path.

## Normalization

Current config:

- `empirical_min = 0.0`
- `empirical_max = 1.0`
- `higher_is_better = true`

This behaves as direct bounded scaling.

## Official Rule

- `official` only when prompt candidates come from explicit annotation fields.
- `weak_supervision` when prompts are derived from caption text.

That distinction is crucial. Caption-derived prompts are useful for analysis, but they are not treated as formal object supervision.

## How To Read It

- High: benchmark objects are recoverable from predicted frames under the official prompt set.
- Low: the video omits, obscures, or corrupts required objects.

## Common Misreads

- This is not a general objectness score.
- The prompt source matters as much as the numeric score.
- Strong scene-level realism does not guarantee strong object-control performance.

## Implementation

- `worldarena/benchmark/metrics/control.py`
- `worldarena/benchmark/annotations.py`
- `worldarena/benchmark/official_runtime.py`
