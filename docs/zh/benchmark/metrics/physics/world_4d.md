# World 4D Metrics

**world_4d** group 比较 predicted world 与 simulator world 在 4D 时空结构上的一致性。它比单帧 depth 更严格，因为它要求几何点和 worldline 轨迹在时间维度上持续合理。

## 输入

| 输入 | 用途 |
| --- | --- |
| `world_gt/points4d.npz`、`pred_world/points4d.npz` | 直接提供 4D 点集时，用于 `chamfer_4d` |
| `world_gt/worldlines.npz`、`pred_world/worldlines.npz` | 直接提供 worldline 轨迹时，用于 worldline 指标 |
| `depth.npz`、`cameras.npz`、`mask.npz` | 当点集或 prediction worldline 缺失时，backend 可用 depth、camera 和 mask lifting / reconstruction 得到替代 artifact |
| `novel_time_depth.npz`、`novel_time_rgb.npz` | optional novel-time reconstruction 指标 |

## 公式

### 4D 点 lifting

如果 `points4d.npz` 缺失，backend 可以从 depth、camera 和 mask 中抽样 lifting。对像素 \(x=(u,v)\)、深度 \(z=D_t(u,v)\)、intrinsics \((f_x,f_y,c_x,c_y)\)：

\[
p^{cam}_{t,u,v}
=
\begin{bmatrix}
\frac{u-c_x}{\max(f_x,\epsilon)}z \\
\frac{v-c_y}{\max(f_y,\epsilon)}z \\
z \\
1
\end{bmatrix}
\]

给定 camera-to-world 矩阵 \(C^{2w}_t\)：

\[
p^{world}_{t,u,v}
=
C^{2w}_t p^{cam}_{t,u,v}
\]

4D 点为：

\[
q_{t,u,v}
=
\left(
p^{world}_{x},
p^{world}_{y},
p^{world}_{z},
\tau_t
\right),
\qquad
\tau_t=
\begin{cases}
0, & T=1\\
\frac{t}{T-1}, & T>1
\end{cases}
\]

实现会按 `point_stride` 抽样，并最多保留 `max_points4d` 个点。

### `chamfer_4d`

设 reference 4D 点集为 \(G=\{g_i\}_{i=1}^{N}\)，prediction 4D 点集为 \(P=\{p_j\}_{j=1}^{M}\)，其中 \(g_i,p_j\in\mathbb{R}^4\)。实现使用非平方欧氏距离：

\[
\mathrm{chamfer\_4d}
=
\frac{1}{2}
\left(
\frac{1}{N}\sum_{i=1}^{N}\min_{1\le j\le M}\|g_i-p_j\|_2
+
\frac{1}{M}\sum_{j=1}^{M}\min_{1\le i\le N}\|p_j-g_i\|_2
\right)
\]

若任一点集为空，backend 不返回该指标。

### worldline 指标

设第 \(n\) 条 reference / prediction worldline 在第 \(t\) 帧的 3D 坐标为：

\[
X^g_{n,t}, X^p_{n,t}\in\mathbb{R}^3
\]

有效标记为：

\[
V_{n,t}
=
\mathrm{valid}_{n,t}
\land
\mathrm{finite}(X^g_{n,t})
\land
\mathrm{finite}(X^p_{n,t})
\]

若没有任何有效点，backend 不返回 worldline 指标。有效点距离：

\[
d_{n,t}=\|X^p_{n,t}-X^g_{n,t}\|_2
\]

\[
\mathrm{worldline\_l2\_error}
=
\frac{\sum_{n,t}V_{n,t}d_{n,t}}{\sum_{n,t}V_{n,t}}
\]

以轨迹起点为参考的位移：

\[
\Delta^g_{n,t}=X^g_{n,t}-X^g_{n,0},\qquad
\Delta^p_{n,t}=X^p_{n,t}-X^p_{n,0}
\]

drift：

\[
\delta_{n,t}=\|\Delta^p_{n,t}-\Delta^g_{n,t}\|_2
\]

\[
\mathrm{worldline\_mean\_drift}
=
\frac{\sum_{n,t}V_{n,t}\delta_{n,t}}{\sum_{n,t}V_{n,t}}
\]

对每条轨迹定义有效帧集合：

\[
I_n=\{t:V_{n,t}=1\}
\]

若 \(I_n\neq\varnothing\)，最终漂移为：

\[
f_n=\delta_{n,\max I_n}
\]

\[
\mathrm{worldline\_final\_drift}
=
\frac{1}{|\{n:I_n\neq\varnothing\}|}
\sum_{n:I_n\neq\varnothing} f_n
\]

默认失败阈值 \(\theta=0.1\)。空轨迹在 fail rate 中按失败计：

\[
\beta_n=
\begin{cases}
1, & I_n=\varnothing\\
\mathbf{1}[f_n>\theta], & I_n\neq\varnothing
\end{cases}
\]

\[
\mathrm{worldline\_fail\_rate}
=
100\cdot
\frac{1}{N_{tracks}}
\sum_{n=1}^{N_{tracks}}
\beta_n
\]

轨迹长度只在非空轨迹上平均：

\[
\mathrm{worldline\_length}
=
100\cdot
\frac{1}{|\{n:I_n\neq\varnothing\}|}
\sum_{n:I_n\neq\varnothing}
\frac{|I_n|}{T}
\]

### optional novel-time reconstruction

对任意 optional novel-time artifact \(Y^g,Y^p\)，先沿时间截到共同长度，并取有限值集合：

\[
V=\{i:\mathrm{finite}(Y^g_i)\land\mathrm{finite}(Y^p_i)\}
\]

令 \(r_i=Y^p_i-Y^g_i\)。则：

\[
\mathrm{l1}
=
\frac{1}{|V|}\sum_{i\in V}|r_i|
\]

\[
\mathrm{charb}
=
\frac{1}{|V|}\sum_{i\in V}\sqrt{r_i^2+\epsilon_c^2},
\qquad \epsilon_c=10^{-3}
\]

字段名前缀决定具体指标：

- `novel_time_depth_l1` / `novel_time_depth_charb`
- `novel_time_rgb_l1` / `novel_time_rgb_charb`

## Core 指标

| 指标 | 计算方式 | 原始分数含义 | 方向 | 当前归一化 |
| --- | --- | --- | --- | --- |
| `chamfer_4d` | 对 reference 与 prediction 的 `(x, y, z, t)` 点集计算双向 Chamfer distance | 整体时空点云是否覆盖同一批 4D 几何位置。 | 越低越好 | `[0.0, 1.0]` |
| `worldline_l2_error` | reference 与 prediction worldline 所有有效轨迹点的 3D 欧氏距离均值 | 每个时间点的平均轨迹位置误差。 | 越低越好 | `[0.0, 1.0]` |
| `worldline_mean_drift` | 比较每条轨迹相对于起点的位移，取有效点平均 | 轨迹形状和累计运动是否与 reference 一致。 | 越低越好 | `[0.0, 1.0]` |
| `worldline_final_drift` | 每条轨迹最后一个有效点的 drift，再取平均 | rollout 末端是否积累了明显轨迹偏差。 | 越低越好 | `[0.0, 1.0]` |
| `worldline_fail_rate` | final drift 超过 evaluator 阈值的轨迹占比 | 轨迹失败率。 | 越低越好 | `[0.0, 1.0]` |
| `worldline_length` | 每条 worldline 在 rollout 中保持有效的平均长度或覆盖率 | 轨迹完整度，反映能否持续跟踪到后续帧。 | 越高越好 | `[0.0, 1.0]` |

`worldline_fail_rate` 和 `worldline_length` 的当前底层 evaluator raw 输出是百分数。当前 WorldArena normalization 表仍写成 `[0, 1]`，自定义 evaluator 或配置需要确保 raw convention 与 normalization 一致。

## Optional 指标

| 指标 | 计算方式 | 原始分数含义 | 方向 |
| --- | --- | --- | --- |
| `novel_time_depth_l1` | prediction 与 reference 的 `novel_time_depth` 做 L1 重建误差 | 未纳入 core rollout 时间点的 depth 外推/插值误差。 | 越低越好 |
| `novel_time_depth_charb` | prediction 与 reference 的 `novel_time_depth` 做 Charbonnier 重建误差 | 对异常值更稳健的 novel-time depth 误差。 | 越低越好 |
| `novel_time_rgb_l1` | prediction 与 reference 的 `novel_time_rgb` 做 L1 重建误差 | 未纳入 core rollout 时间点的 RGB 重建误差。 | 越低越好 |
| `novel_time_rgb_charb` | prediction 与 reference 的 `novel_time_rgb` 做 Charbonnier 重建误差 | 对异常值更稳健的 novel-time RGB 误差。 | 越低越好 |

启用 optional novel-time 指标时，`world_gt/` 和 `pred_world/` 都必须提供对应 artifact，并在自定义配置中声明 normalization。

## 适合解读的问题

- 预测是否形成了连续、可追踪的 4D 世界结构。
- 物体轨迹是否随时间持续接近 simulator reference。
- 长 rollout 末端是否发生累计漂移。

## 不适合单独解读的问题

- 4D 点集正确不保证 prompt、材质或视觉质量正确。
- lifting / reconstruction 依赖 depth、camera、mask artifact 的质量。
- `chamfer_4d` 是集合距离，不直接保留一一对应的物体身份。

## 常见失败形态

- 预测 depth 或 camera 错误导致 lifted points 偏移，`chamfer_4d` 上升。
- worldline 中途丢失或无效，`worldline_length` 下降。
- 轨迹形状正确但整体 scale 错，`worldline_l2_error` 和 drift 指标都会变差。
