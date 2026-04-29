# Quality

The `quality` dimension covers style, perceptual quality, and reconstruction-style fidelity. It asks whether outputs are visually stable, clear, natural, and, when paired references exist, reconstructively similar.

## Scope

| Layer | Metrics |
| --- | --- |
| style and perceptual quality | [`style_consistency`](../style_consistency.md), [`perceptual_quality`](../perceptual_quality.md) |
| paired reconstruction | [`psnr`](../psnr.md), [`ssim`](../ssim.md) |
| Long-Sequence quality diagnostics | [`long_sequence_aesthetic_quality`](../long_sequence.md), [`long_sequence_imaging_quality`](../long_sequence.md) |

## Reading Notes

- `style_consistency` and `perceptual_quality` are core quality readings in the formal suites.
- `psnr` and `ssim` are paired-frame reconstruction metrics and are better treated as diagnostics than replacements for semantic or physical metrics.
- Quality metrics alone cannot establish control, memory, or physics correctness.
