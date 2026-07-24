# Object Detection

**object_detection** measures whether requested scene objects can be recovered from prediction frames under the benchmark's prompt-matching contract.

## What It Measures

- recoverability of required objects from generated frames
- object presence under an open-vocabulary detector
- not box accuracy, not full scene understanding, and not object motion correctness

## Detailed Algorithm

1. WorldArena derives `prompt_candidates` from the annotation directory.
2. Prompt extraction accepts only explicit structured object fields in the annotation payload.
   Caption-derived noun phrases are useful for data auditing but are not scored by this metric.
3. The metric builds `prompt_string = "scene, " + ", ".join(prompt_candidates)`.
4. The wrapped third-party metric runs `Grounding DINO` on every decoded frame with box threshold `0.4` and text threshold `0.4`.
5. Grounding DINO returns phrases for the detected regions.
6. Matching is scored in two passes:
    - exact phrase matches against the remaining prompt list
    - partial matches based on noun overlap after lightweight spaCy parsing
7. The frame score is `frame_score = min(matched_prompt_count / len(prompt_candidates), 1.0)`.
8. The raw metric is `raw = mean frame_score` across all frames.
9. Current normalization is direct bounded scaling in `[0, 1]`.

## Inputs And Outputs


| Item              | Current implementation                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Prediction input  | full decoded prediction video                                                                                            |
| Reference input   | none                                                                                                                     |
| Annotation input  | structured object labels from the annotation directory                                                                   |
| Raw output        | one scalar object-recoverability score in `[0, 1]`                                                                       |
| Normalized output | same scalar in `[0, 1]`                                                                                                  |
| Details payload   | `prompt_source`, `prompt_candidates`, `prompt_string`, `frame_count`, `native_score`                                     |
| Failure mode      | returns `not_applicable` if explicit object prompts are unavailable; returns `raw = null`, `normalized = null`, and an error string if the detector runtime fails |


## Integration Readiness


| Item                | Verdict                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Current aggregation | computed as `official` only when prompts come from explicit annotations; caption-derived prompts are `not_applicable` |
| Publication verdict | `Supporting only`                                                                                                |


- Why it is not stronger: there is no box-level ground truth comparison. The metric only asks whether an open-vocabulary detector can recover requested object phrases from the generated frames.
- The matching stage is heuristic by design: it mixes exact string matching and noun-overlap matching, which is useful but not equivalent to a rigorous detection benchmark.
- Caption-derived prompts are excluded from the scoring path because they cannot support a formal object-control claim.
- Recommended use: acceptable as a supporting object-recoverability signal, especially when `prompt_source = annotation_explicit`, but do not present it as a strict detection metric.
