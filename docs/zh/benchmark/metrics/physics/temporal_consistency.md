# Temporal Consistency Metrics

**temporal_consistency** group 检查相邻帧之间的几何、光流和 RGB 自一致性。它关注预测是否能形成连续的时间演化，而不是每一帧单独看起来是否合理。

## 输入

| 输入 | 用途 |
| --- | --- |
| `world_gt/depth.npz`、`pred_world/depth.npz` | 用于 depth warp 指标 |
| `world_gt/flow.npz`、`pred_world/flow.npz` | 用于光流误差与 warp |
| `world_gt/occlusion.npz`、`world_gt/mask.npz` | 可选有效区域，用来排除遮挡或无效 flow 区域 |
| prediction video | 用于 `rgb_warp_charb` 和可选 `rgb_warp_lpips` |

## 公式

实现使用双线性采样 warp。给定单帧张量 \(X_t\) 和 flow \(F_t(x)=(u_t(x), v_t(x))\)，warp 定义为：

\[
W(X_t,F_t)(x)=X_t(x+F_t(x))
\]

其中越界采样使用 border padding。Charbonnier penalty 为：

\[
\rho_\epsilon(z)=\sqrt{z^2+\epsilon_c^2}, \qquad \epsilon_c=10^{-3}
\]

### `depth_warp_l1` / `depth_warp_charb`

设 reference flow 为 \(F^g_t\)，reference depth 为 \(D^g_t\)，prediction depth 为 \(D^p_t\)。实现使用：

\[
S=\min(T_{F^g}, T_{D^p}-1, T_{D^g}-1)
\]

对每个 \(t=1,\ldots,S\)，有效像素集合为：

\[
V_t =
\left\{x:
\mathrm{finite}(W(D^p_t,F^g_t)(x)) \land
\mathrm{finite}(D^g_{t+1,x}) \land
D^g_{t+1,x}>\epsilon \land A_{t,x}
\right\}
\]

其中 \(A_{t,x}\) 来自 `world_gt/occlusion.npz` 的非遮挡区域，或来自相邻 mask 的交集；没有这些 artifact 时 \(A_{t,x}=1\)。令：

\[
e_{t,x}=W(D^p_t,F^g_t)(x)-D^g_{t+1,x}
\]

只对 \(V_t\neq\varnothing\) 的时间步求平均，记这些时间步集合为 \(\mathcal{T}\)：

\[
\mathrm{depth\_warp\_l1}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{1}{|V_t|}
\sum_{x\in V_t}|e_{t,x}|
\]

\[
\mathrm{depth\_warp\_charb}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{1}{|V_t|}
\sum_{x\in V_t}\rho_\epsilon(e_{t,x})
\]

### `rgb_warp_charb`

设 prediction RGB 为 \(P_t \in [0,1]^{C\times H\times W}\)，prediction flow 为 \(F^p_t\)。对：

\[
S=\min(T_{F^p}, T_P-1)
\]

定义：

\[
E_{t,c,x}=W(P_t,F^p_t)_{c,x}-P_{t+1,c,x}
\]

若存在有效 mask \(B_t\)，它会扩展到所有 channel；否则 \(B_t=\Omega\)。跳过 \(B_t=\varnothing\) 的时间步，记剩余集合为 \(\mathcal{T}\)：

\[
\mathrm{rgb\_warp\_charb}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{1}{C|B_t|}
\sum_{c=1}^{C}\sum_{x\in B_t}
\rho_\epsilon(E_{t,c,x})
\]

### `epe` / `fl_all` / `one_px_out`

设 reference flow 与 prediction flow 为 \(F^g_{t,x}, F^p_{t,x}\in\mathbb{R}^2\)。有效集合：

\[
V=\{(t,x): \mathrm{finite}(F^g_{t,x}) \land \mathrm{finite}(F^p_{t,x}) \land A_{t,x}\}
\]

每个有效点的 endpoint error 和 reference flow magnitude：

\[
e_{t,x}=\|F^p_{t,x}-F^g_{t,x}\|_2,\qquad
m_{t,x}=\|F^g_{t,x}\|_2
\]

\[
\mathrm{epe}
=
\frac{1}{|V|}
\sum_{(t,x)\in V} e_{t,x}
\]

outlier 指示函数：

\[
o_{t,x}
=
\mathbf{1}
\left[
e_{t,x}>3.0
\land
\frac{e_{t,x}}{\max(m_{t,x},\epsilon)}>0.05
\right]
\]

当前 evaluator 的 raw 输出是百分数：

\[
\mathrm{fl\_all}
=
100\cdot
\frac{1}{|V|}
\sum_{(t,x)\in V} o_{t,x}
\]

\[
\mathrm{one\_px\_out}
=
100\cdot
\frac{1}{|V|}
\sum_{(t,x)\in V}
\mathbf{1}[e_{t,x}>1.0]
\]

## Core 指标

| 指标 | 计算方式 | 原始分数含义 | 方向 | 当前归一化 |
| --- | --- | --- | --- | --- |
| `depth_warp_l1` | 使用 reference flow 把 `pred_depth[t]` warp 到下一帧，再与 `gt_depth[t+1]` 比较 L1 误差 | 跨帧深度是否能沿 simulator 运动场演化到正确的下一帧几何。 | 越低越好 | `[0.0, 1.0]` |
| `depth_warp_charb` | 与 `depth_warp_l1` 相同，但使用 Charbonnier penalty | 对异常值更稳健的跨帧深度 warp 误差。 | 越低越好 | `[0.0, 1.0]` |
| `rgb_warp_charb` | 使用 prediction flow 把预测 RGB 帧 `t` warp 到 `t+1`，再与预测 RGB 帧 `t+1` 比较 | 预测视频和预测 flow 的自一致性；不直接比较 reference RGB。 | 越低越好 | `[0.0, 0.5]` |
| `epe` | `pred_flow` 与 `gt_flow` 的 endpoint error | 平均光流向量误差，单位是像素位移。 | 越低越好 | `[0.0, 10.0]` |
| `fl_all` | 满足 flow outlier 条件的有效像素比例或百分率 | KITTI 风格 flow outlier rate，通常要求 EPE 大于 3px 且相对误差大于 5%。 | 越低越好 | `[0.0, 1.0]` |
| `one_px_out` | EPE 大于 1px 的有效像素比例或百分率 | 更严格的一像素 outlier rate，用来暴露小位移和边界区域不稳定。 | 越低越好 | `[0.0, 1.0]` |

`fl_all` 和 `one_px_out` 的当前底层 evaluator raw 输出是百分数。当前 WorldArena normalization 表仍写成 `[0, 1]`，自定义 evaluator 或配置需要确保 raw convention 与 normalization 一致。

## Optional 指标

| 指标 | 计算方式 | 原始分数含义 | 方向 |
| --- | --- | --- | --- |
| `rgb_warp_lpips` | 与 `rgb_warp_charb` 相同的 prediction-flow warp 对齐目标，但使用 LPIPS 感知距离 | 预测视频内部的感知级时序一致性。 | 越低越好 |

\[
\mathrm{rgb\_warp\_lpips}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\mathrm{LPIPS}\left(
W(P_t,F^p_t)\odot B_t,\,
P_{t+1}\odot B_t
\right)
\]

启用 `rgb_warp_lpips` 时，需要 LPIPS runtime dependency，并在自定义配置中声明 normalization。

## 适合解读的问题

- 相邻帧的深度和光流是否连续。
- 预测 motion field 是否能解释自己的 RGB 时间演化。
- 运动边界、遮挡区域和快速运动区域是否稳定。

## 不适合单独解读的问题

- `rgb_warp_charb` 是 prediction self-consistency，流和 RGB 一起错时也可能得到较好分数。
- 光流指标依赖 flow artifact 质量。
- 这组指标不直接判断 prompt 是否被遵循。

## 常见失败形态

- 物体 teleport 或跳帧，导致 depth warp 和 flow outlier 上升。
- 预测 flow 与 RGB 帧不匹配，导致 `rgb_warp_charb` 上升。
- 遮挡处理不一致，导致边界区域 EPE 或 one-pixel outlier 上升。
