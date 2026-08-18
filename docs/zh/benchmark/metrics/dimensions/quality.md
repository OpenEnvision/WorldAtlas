# Quality

`quality` 维度覆盖风格、感知质量、低层外观稳定性与重建式保真。它关注输出是否视觉上稳定、清晰、自然，以及在 paired reference 存在时是否有较好的重建相似性，并将维度汇总结果写入 `per_dimension_summary.json`。

### 覆盖范围

本维度按视觉质量的不同层面组织指标。风格与感知质量由 [`style_consistency`](../style_consistency.md)、[`perceptual_quality`](../perceptual_quality.md)、[`image_quality`](../image_quality.md) 和 [`hpsv3_norm`](../hpsv3_norm.md) 共同刻画。

外观稳定性由 [`brightness_distribution_consistency`](../brightness_consistency.md)、[`color_temperature_constraint`](../color_temperature_consistency.md) 和 [`sharpness_retention`](../sharpness_retention.md) 衡量，分别关注亮度分布、色温约束与清晰度保持。Paired reconstruction 指标包括 [`psnr`](../psnr.md) 和 [`ssim`](../ssim.md)，用于在存在配对参考时衡量重建相似性。Long-Sequence quality 则由 [`long_sequence_aesthetic_quality`](../long_sequence.md) 和 [`long_sequence_imaging_quality`](../long_sequence.md) 补充长序列场景下的审美与成像质量。

### 解读边界

七个完整的正式 `image_static` quality 指标为：`sharpness_retention`、`perceptual_quality`、`style_consistency`、`brightness_distribution_consistency`、`color_temperature_constraint`、`image_quality` 和 `hpsv3_norm`。七项全部进入 `per_metric_summary`，并等权计入 `quality` 维度总分；旧别名 `brightness_consistency` 与 `color_temperature_consistency` 仍保留为 diagnostic。

这些低层指标受 iWorld-Bench quality 槽位启发，但不是对未公开 iWorld-Bench 官方代码的复现。`psnr` / `ssim` 是 paired-frame 重建式指标，更适合作为补充读数，不应替代语义或物理指标。quality 指标也无法单独说明 control、memory 或 physics 是否正确。
