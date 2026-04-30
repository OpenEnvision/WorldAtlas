# 清晰度保持

`sharpness_retention` 衡量生成视频在初始帧之后是否保持可见纹理清晰度。

## 衡量内容

- 随时间积累的模糊
- rollout 开始后边缘能量衰减
- 启用默认后端时，通过 BRISQUE gate 过滤高频伪影

该指标参考 iWorld-Bench 的 Sharpness Retention 指标槽位；公开说明提到了 Tenengrad 和 BRISQUE-triggered circuit breaker。由于官方实现尚未公开，这里只作为 WorldArena 本地诊断指标。

## 计算流程

1. 读取预测视频的采样帧。
2. 使用 Sobel 梯度能量计算每帧的 Tenengrad 清晰度。
3. 用最前面若干 baseline 帧的 Tenengrad 中位数作为基线。
4. 后续帧的分数为 `retention = clamp(tenengrad_t / baseline, 0, 1)`。
5. 基础清晰度保持分数是后续帧 retention 的平均值。
6. 默认 `tenengrad_brisque` 后端还会在相同帧上计算 BRISQUE。
7. 当平均 BRISQUE 高于配置的 bad 阈值时，gate 会把分数压低；如果 BRISQUE 后端缺失，会显式返回错误，不做静默降级。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 预测视频采样帧 |
| 参考输入 | 无 |
| 标注输入 | 无 |
| 原始输出 | `[0, 1]` 标量；越高表示清晰度保持越好 |
| 归一化输出 | 默认配置下为恒等映射 |
| 细节字段 | Tenengrad 分数、baseline、retention 分数、BRISQUE 分数、BRISQUE gate |
| 失败模式 | 少于两帧、baseline 纹理过低或配置的 BRISQUE 后端不可用时，返回 `raw = null`、`normalized = null` 和错误信息 |

## 集成状态

| 项目 | 判断 |
| --- | --- |
| 当前聚合 | 仅 `diagnostic` |
| 论文报告定位 | 辅助诊断 |

- 作用：能发现语义指标不一定敏感的模糊退化。
- 局限：高边缘能量也可能来自伪影，因此默认后端会用 BRISQUE gate 约束 Tenengrad。
- 建议：作为 quality 维度诊断，与 `perceptual_quality` 一起阅读，不作为主分数。
