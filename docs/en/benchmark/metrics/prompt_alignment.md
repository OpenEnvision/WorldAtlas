# Prompt Alignment

`prompt_alignment` measures whether the predicted video matches `sample.prompt_target` at the benchmark's protocol anchor frames. All WorldArena formal suites evaluate video predictions; `image_static` and `image_dynamic` are image-conditioned video rollouts, not still-image outputs.

## What It Measures

- semantic agreement between the generated content and the target instruction
- anchor-frame instruction following, not full-video temporal reasoning
- content correctness, not realism, smoothness, or reference copying

## Detailed Algorithm

1. The predicted video is first downsampled to at most `benchmark.motion_frame_count` uniformly spaced frames, and the current official protocol `segment_random` then picks one protocol frame per segment from that sampled set. Image suites may use a single reference image as conditioning, but the model prediction must still be a video.
2. For each prediction anchor frame `f_i`, WorldArena calls `torchmetrics.multimodal.CLIPScore` with model `openai/clip-vit-base-patch16` and text `sample.prompt_target`.
3. The native CLIPScore `s_i` is converted into a bounded frame score with `frame_score_i = clamp(s_i / 100, 0, 1)`.
4. The raw metric is `raw = mean_i frame_score_i`.
5. Benchmark normalization is identity in the current config because `lower = 0`, `upper = 1`, and `higher_is_better = true`, so `normalized = raw`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | video prediction only; prediction video anchor frames |
| Reference input | none |
| Annotation input | `sample.prompt_target`, `sample.has_instruction`, `sample.conditioning_strategy` |
| Raw output | a scalar mean anchor score in `[0, 1]` |
| Normalized output | same scalar in `[0, 1]` |
| Details payload | `frame_scores`, `native_frame_scores`, `native_score_range`, `frame_indices`, `sampled_frame_indices` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if the prediction is not a video or the CLIPScore backend is unavailable |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | computed as `official` only if `sample.has_instruction` is true and `sample.conditioning_strategy != "self"`; otherwise `not_applicable` |
| Publication verdict | `Main-table ready` |

- Why it is relatively strong: the algorithm is implemented directly in this repo, the score transformation is simple, and tests already cover scaling and eligibility routing.
- Prompt text must describe the target video result. Image-suite `prompt_target` values should also be written as video-rollout instructions, not image-generation instructions.
- Samples without explicit instruction supervision, or self-conditioned samples, receive no score because that value cannot support a strict prompt-following claim.
- Why it still should not stand alone: CLIPScore is a semantic proxy, not human judgment, and anchor-frame evaluation can miss long-horizon temporal failures.
- Recommended use: keep it as a main semantic metric, but pair it with quality and control metrics that test different failure modes.
