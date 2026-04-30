# Quality

The `quality` dimension covers style, perceptual quality, low-level appearance stability, and reconstruction-style fidelity. It asks whether outputs are visually stable, clear, natural, and, when paired references exist, reconstructively similar.

## Scope

| Layer | Metrics |
| --- | --- |
| style and perceptual quality | [`style_consistency`](../style_consistency.md), [`perceptual_quality`](../perceptual_quality.md) |
| appearance stability diagnostics | [`brightness_consistency`](../brightness_consistency.md), [`color_temperature_consistency`](../color_temperature_consistency.md), [`sharpness_retention`](../sharpness_retention.md) |
| paired reconstruction | [`psnr`](../psnr.md), [`ssim`](../ssim.md) |
| Long-Sequence quality diagnostics | [`long_sequence_aesthetic_quality`](../long_sequence.md), [`long_sequence_imaging_quality`](../long_sequence.md) |

## Reading Notes

- `style_consistency` and `perceptual_quality` are core quality readings in the formal suites.
- `brightness_consistency`, `color_temperature_consistency`, and `sharpness_retention` are low-level diagnostics inspired by iWorld-Bench quality slots; they are not official reproductions of unpublished iWorld-Bench code.
- `psnr` and `ssim` are paired-frame reconstruction metrics and are better treated as diagnostics than replacements for semantic or physical metrics.
- Quality metrics alone cannot establish control, memory, or physics correctness.
