# Semantic Alignment

`semantic_alignment` measures whether the prediction matches the annotated reference content, not merely
the benchmark prompt.

## What It Measures

- semantic closeness to the reference video's caption-side content
- reference-conditioned content agreement, not prompt following
- a diagnostic proxy for "does this look like the same scene content?"

## Detailed Algorithm

1. WorldArena builds a reference text string from `caption.json` under `sample.annotation_path`. If
   instruction labels exist, it appends them as auxiliary context. If annotation text is missing, it
   falls back to `sample.prompt_current` or `sample.prompt_target`.
2. The prediction's sampled frames are encoded with the CLIP image encoder
   `openai/clip-vit-base-patch16`.
3. The reference text is encoded with the matching CLIP text encoder.
4. For each sampled frame, WorldArena computes cosine similarity to the reference text embedding and
   maps it into `[0, 1]` with `(cos + 1) / 2`.
5. The raw metric is the mean frame score. Current normalization is identity-like in the default
   config because the score is already bounded in `[0, 1]`.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled prediction frames |
| Reference input | reference annotation text, not reference pixels |
| Annotation input | `caption.json`, optional `instructions.json`, fallback prompt text |
| Raw output | one scalar mean semantic score in `[0, 1]` |
| Normalized output | the same bounded scalar under the default config |
| Details payload | `reference_text_source`, `reference_text`, `frame_scores`, frame index metadata |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string if CLIP is unavailable or no reference text can be built |

## Integration Readiness

| Item | Verdict |
| --- | --- |
| Current aggregation | `diagnostic` only |
| Publication verdict | `Supporting only` |

- Why it is useful: unlike `prompt_alignment`, it is anchored to reference-side content.
- Why it is limited: it still depends on annotation text quality and CLIP as a semantic proxy.
- Recommended use: pair it with prompt-following and quality metrics to separate prompt mismatch from
  reference-content mismatch.
