# Quality

`quality` 维度覆盖风格、感知质量、低层外观稳定性与重建式保真。它关注输出是否视觉上稳定、清晰、自然，以及在 paired reference 存在时是否有较好的重建相似性。

## 覆盖范围

| 层面 | 指标 |
| --- | --- |
| 风格与感知质量 | [`style_consistency`](../style_consistency.md)、[`perceptual_quality`](../perceptual_quality.md) |
| 外观稳定性诊断 | [`brightness_consistency`](../brightness_consistency.md)、[`color_temperature_consistency`](../color_temperature_consistency.md)、[`sharpness_retention`](../sharpness_retention.md) |
| paired reconstruction | [`psnr`](../psnr.md)、[`ssim`](../ssim.md) |
| Long-Sequence quality 诊断 | [`long_sequence_aesthetic_quality`](../long_sequence.md)、[`long_sequence_imaging_quality`](../long_sequence.md) |

## 解读重点

- `style_consistency` 和 `perceptual_quality` 是 formal suite 的核心质量读数。
- `brightness_consistency`、`color_temperature_consistency`、`sharpness_retention` 是受 iWorld-Bench quality 槽位启发的低层诊断指标；它们不是对未公开 iWorld-Bench 官方代码的复现。
- `psnr` / `ssim` 是 paired-frame 重建式指标，更适合作 diagnostic，不应替代语义或物理指标。
- quality 指标无法单独说明 control、memory、physics 是否正确。
