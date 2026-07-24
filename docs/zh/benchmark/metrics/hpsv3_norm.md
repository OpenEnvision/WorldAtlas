# HPSv3-Norm

HPSv3-Norm 采用 HPSv3 奖励模型度量生成视频的**人类偏好对齐视觉质量**。该指标**仅需 prediction**：各评测帧独立打分，无需与 reference rollout 配对。

`hpsv3_norm` 是正式的 `image_static` quality 指标，同时进入 `per_metric_summary` 与七项等权 quality 总分。

WorldArena 从 prediction 视频中均匀子采样至多 20 帧，默认以 HPSv3 官方评测文档中提供的批大小（`batch_size: 8`）推理；这也是 8×A100 视频评测中实测的吞吐折中点。批处理只改变执行方式：每帧仍独立获得一个奖励；为保证可比性，同一次横向评测应使用相同的批大小。对帧 $x_i$，HPSv3 推理器输出原始奖励 $r_i$。每视频平均原始奖励为

\[
\bar{r}
=
\frac{1}{N}\sum_{i=1}^{N} r_i
\tag{1}
\]

设固定归一化分位数为 $p_1$、$p_{99}$（默认 $p_1 = 5.21$，$p_{99} = 8.66$）。$[0, 100]$ 上的归一化得分为

\[
S_{\mathrm{HPS}}
=
\mathrm{clip}\!\left(
\frac{\bar{r} - p_1}{p_{99} - p_1} \times 100,\,
0,\,
100
\right)
\tag{2}
\]

WorldArena 将指标 raw 值存为 $[0, 1]$ 区间内的 $S_{\mathrm{HPS}} / 100$。得分越高表示与人类偏好对齐程度越强。若 HPSv3 backend 或 checkpoint 不可用，指标记为 `not_applicable`。

## 参考文献 {#reference}

!!! quote "HPSv3 — Human Preference Score"

    基于人类偏好监督的视觉-语言奖励模型；WorldArena 对每视频平均原始奖励施加线性分位数归一化。
