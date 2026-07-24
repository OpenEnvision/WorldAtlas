# Geometry 3D Metrics

**geometry_3d** group 比较 predicted world 导出的 depth 与 simulator 导出的 depth。它关注生成视频背后的三维几何是否合理，而不只是 RGB 帧是否相似。

## 输入

| 输入 | 用途 |
| --- | --- |
| `world_gt/depth.npz` | simulator reference depth |
| `pred_world/depth.npz` | 从生成视频或 predicted world pipeline 得到的 prediction depth |
| `world_gt/mask.npz` | 可选有效区域约束，用来只在目标区域或有效区域上计算深度误差 |

所有指标只在有限、正深度且有效 mask 覆盖的位置上计算。若没有任何有效像素，backend 会缺失这些指标，进而触发 required metric failure。

## 公式

设 reference depth 为 \(D^g\)，prediction depth 为 \(D^p\)。二者先沿时间截到共同长度。有效集合定义为：

\[
V =
\left\{(t,x):
\mathrm{finite}(D^g_{t,x}) \land
\mathrm{finite}(D^p_{t,x}) \land
D^g_{t,x}>\epsilon \land
D^p_{t,x}>\epsilon \land
A_{t,x}
\right\}
\]

其中 \(A_{t,x}\) 是可选 valid mask；没有额外 mask 时 \(A_{t,x}=1\)。令 \(N=|V|\)，\(\epsilon=10^{-6}\)。如果 \(N=0\)，backend 不返回这些 depth 指标。

### `abs_rel`

\[
\mathrm{abs\_rel}
= \frac{1}{N}\sum_{(t,x)\in V}
\frac{|D^p_{t,x}-D^g_{t,x}|}{\max(D^g_{t,x},\epsilon)}
\]

### `rmse`

\[
\mathrm{rmse}
=
\sqrt{
\frac{1}{N}\sum_{(t,x)\in V}
\left(D^p_{t,x}-D^g_{t,x}\right)^2
}
\]

### `delta1` / `delta2` / `delta3`

先定义每个有效点的相对比例误差：

\[
r_{t,x}
=
\max\left(
\frac{D^p_{t,x}}{\max(D^g_{t,x},\epsilon)},
\frac{D^g_{t,x}}{\max(D^p_{t,x},\epsilon)}
\right)
\]

对 \(k \in \{1,2,3\}\)：

\[
\delta_k
=
\frac{1}{N}
\sum_{(t,x)\in V}
\mathbf{1}\left[r_{t,x}<1.25^k\right]
\]

对应字段为：

\[
\mathrm{delta1}=\delta_1,\qquad
\mathrm{delta2}=\delta_2,\qquad
\mathrm{delta3}=\delta_3
\]

## 指标

| 指标 | 计算方式 | 原始分数含义 | 方向 | 当前归一化 |
| --- | --- | --- | --- | --- |
| `abs_rel` | `mean(abs(pred_depth - gt_depth) / gt_depth)` | 相对深度误差。近处物体的尺度错误会被放大。 | 越低越好 | `[0.0, 1.0]` |
| `rmse` | `sqrt(mean((pred_depth - gt_depth)^2))` | 绝对深度均方根误差。少量很大的几何错误会显著抬高分数。 | 越低越好 | `[0.0, 5.0]` |
| `delta1` | `max(pred/gt, gt/pred) < 1.25` 的有效像素比例 | 严格深度命中率，约等于 25% 相对误差范围内的比例。 | 越高越好 | `[0.0, 1.0]` |
| `delta2` | `max(pred/gt, gt/pred) < 1.25^2` 的有效像素比例 | 中等宽松的深度命中率。 | 越高越好 | `[0.0, 1.0]` |
| `delta3` | `max(pred/gt, gt/pred) < 1.25^3` 的有效像素比例 | 最宽松的深度命中率，主要暴露大范围尺度崩坏。 | 越高越好 | `[0.0, 1.0]` |

## 适合解读的问题

- 预测主体和场景是否保持正确的相对深度。
- 生成视频是否产生了明显的几何尺度错误。
- predicted world 的 depth 是否能支持后续 4D / worldline 指标。

## 不适合单独解读的问题

- 深度准确不保证 RGB 外观、物体身份或运动因果正确。
- depth metric 对 predicted world backend 的深度估计质量敏感。
- 如果相机或 scale convention 不一致，`abs_rel` 和 `rmse` 会被系统性放大。

## 常见失败形态

- 预测主体深度整体过近或过远，`abs_rel` 和 `rmse` 上升。
- 局部几何破碎或深度空洞导致 `delta1/2/3` 下降。
- depth artifact 缺失或没有有效 mask 区域，导致 required metric missing。
