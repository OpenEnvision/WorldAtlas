# Trajectory Dynamics Metrics

`trajectory_dynamics` group 用完整 rollout 评估 simulator-backed inertia 行为。它面向不能只看最后一帧的场景：模型可能刚好落在正确终点附近，但中间过程出现明显非惯性运动。

## 输入

| 输入 | 用途 |
| --- | --- |
| `world_gt/worldlines.npz` | reference tracked points，包含 `xyz`、`valid`，通常还包含 `pixels` |
| `pred_world/worldlines.npz` | prediction tracked points；缺失时，WorldArena 会用 `pred_world/depth.npz`、`pred_world/cameras.npz` 和 reference worldline pixels 重建 |
| `physics_spec.trajectory_metric.coordinate_space` | 可选坐标约定；默认是 `world_xyz`，`normalized_pixels` 表示用像素轨迹除以样本宽高 |

默认 simulator 实现使用 `world_xyz`，因为 PSIVG artifact 契约直接提供 simulator worldline。自定义 manifest 可以设置 `coordinate_space: normalized_pixels` 来匹配 2D 归一化像素公式，但这要求 reference 和 prediction worldline artifact 都包含 `pixels`。

## 公式

设第 \(n\) 条 track 在第 \(t\) 帧的 reference / prediction 位置为 \(p^g_{n,t}\) 和 \(p^p_{n,t}\)。有效 mask 要求两边 track 都有效且坐标有限。

### `trajectory_rmse`

\[
\mathrm{trajectory\_rmse}
=
\sqrt{
\frac{1}{|V|}
\sum_{(n,t)\in V}
\|p^g_{n,t}-p^p_{n,t}\|_2^2
}
\]

越低越好。

### `final_position_error`

对每条 final frame 有效的轨迹，定义 reference path length：

\[
L^g_n =
\sum_{t=1}^{T-1}
\|p^g_{n,t+1}-p^g_{n,t}\|_2
\]

若轨迹近似静止，\(L^g_n \approx 0\)，该比例没有定义，当前实现会跳过这条轨迹。指标平均：

\[
\mathrm{final\_position\_error}
=
\frac{1}{|\mathcal{N}|}
\sum_{n\in\mathcal{N}}
\frac{\|p^g_{n,T}-p^p_{n,T}\|_2}{L^g_n}
\]

越低越好。

### `speed_similarity`

速度是一阶差分：

\[
v^g_{n,t}=p^g_{n,t}-p^g_{n,t-1},\qquad
v^p_{n,t}=p^p_{n,t}-p^p_{n,t-1}
\]

分数是在有效相邻帧对上的平均 cosine similarity：

\[
\mathrm{speed\_similarity}
=
\frac{1}{|\mathcal{V}|}
\sum_{(n,t)\in\mathcal{V}}
\frac{v^g_{n,t}\cdot v^p_{n,t}}
{\|v^g_{n,t}\|_2\|v^p_{n,t}\|_2}
\]

当两边速度都接近零时，WorldArena 把该 pair 记为 similarity `1`；只有一边接近零时记为 `0`。

### `acceleration_similarity`

加速度是速度差分：

\[
a^g_{n,t}=v^g_{n,t}-v^g_{n,t-1},\qquad
a^p_{n,t}=v^p_{n,t}-v^p_{n,t-1}
\]

指标是在有效加速度 pair 上的平均 cosine similarity。它沿用 `speed_similarity` 的近零规则，因此正确的匀速轨迹会得到高加速度分，而不是变成 undefined。

### `directional_consistency`

对每条首帧和末帧都有效的轨迹：

\[
d^g_n=p^g_{n,T}-p^g_{n,1},\qquad
d^p_n=p^p_{n,T}-p^p_{n,1}
\]

设 \(\theta_n\) 是两个方向向量的夹角，单位为度。单条轨迹分数为：

\[
S^{dir}_n=\frac{180-\theta_n}{180}
\]

最终指标在有效轨迹上平均。越高越好。

## Core 指标

| 指标 | 原始分数含义 | 方向 | 当前归一化 |
| --- | --- | --- | --- |
| `trajectory_rmse` | predicted tracked-point path 与 reference path 的全局偏差 | 越低越好 | `[0.0, 1.0]` |
| `final_position_error` | 末端误差除以 reference path length | 越低越好 | `[0.0, 1.0]` |
| `speed_similarity` | 局部速度方向一致性 | 越高越好 | `[-1.0, 1.0]` |
| `acceleration_similarity` | 局部加速度模式一致性 | 越高越好 | `[-1.0, 1.0]` |
| `directional_consistency` | start-to-end 总体运动方向一致性 | 越高越好 | `[0.0, 1.0]` |

## 失败模式

- 缺少 `world_gt/worldlines.npz` 或 `pred_world/worldlines.npz`。
- `coordinate_space` 设置为 `normalized_pixels`，但 worldline artifact 缺少 `pixels`。
- 轨迹 shape 不一致，或没有任何有效 tracked point。
- `final_position_error` 没有可用的非静止轨迹。
