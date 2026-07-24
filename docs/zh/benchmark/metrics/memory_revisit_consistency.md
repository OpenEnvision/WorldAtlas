# 记忆回访一致性

**memory_revisit_consistency** 用来评估 360 度回访类 rollout：模型中途必须真的离开初始视角，最后还要回到同一个生成场景。

## 它测什么

- 首帧与末帧是否闭环一致
- 中途是否发生足够视角变化，避免“完全不动但闭环满分”
- 适合 360 orbit / revisit / loop-closing 类样本；不适合普通开放式视频

## 当前算法

1. 读取 prediction 的协议采样帧；少于 2 帧则返回空分数。
2. 取 `start = frames[0]`，`end = frames[-1]`，把末帧 resize 到首帧尺寸。
3. 用 DINOv2 提取全部采样帧特征。
4. 计算闭环三分量：
    - `dino_revisit = sim_dino(start, end)`
    - `ssim_revisit = SSIM(start, end)`
    - `mse_revisit = 1 - clamp(MSE(start, end) / mse_tau, 0, 1)`，默认 `mse_tau = 0.08`
5. 闭环分数是加权平均：默认 `dino = 0.55`、`ssim = 0.25`、`mse = 0.20`。
6. 取 25%、50%、75% 三个中途帧，计算它们相对首帧的 DINO displacement：`1 - sim_dino(start, frame_i)`。
7. 中途进展分数：`progress_score = clamp((mean_displacement - progress_lower) / (progress_upper - progress_lower), 0, 1)`，默认区间 `[0.03, 0.18]`。
8. 最终原始分数：`raw = closure_score * ((1 - gate) + gate * progress_score)`，默认 `gate = 0.35`。

这个门控让“完全静止”的视频即使首末帧完全一致，也只能拿到约 `0.65`。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | prediction video 的协议采样帧 |
| 参考输入 | 不使用 |
| 标注输入 | 不使用 |
| 后端 | `dinov2` |
| 原始输出 | `[0,1]` 闭环一致性分数 |
| 归一化输出 | 默认恒等映射 |
| 细节字段 | `dino_revisit`、`ssim_revisit`、`mse`、`mse_revisit`、`closure_score`、`quarter_indices`、`quarter_dino_similarities`、`mean_quarter_displacement`、`progress_score`、`progress_gate_weight` |

## 聚合状态

| 项目 | 结论 |
| --- | --- |
| 当前 suite | `experimental` |
| 当前聚合 | 永远 `diagnostic` |
| 维度 | `long_sequence` |
| 论文用法 | 适合作为 memory / loop-closing 附录指标；不建议进主表 |

关键限制：它没有使用几何真值，也不验证相机角度是否真的完成 360 度；它只证明“视觉上离开过，并且末帧像首帧”。
