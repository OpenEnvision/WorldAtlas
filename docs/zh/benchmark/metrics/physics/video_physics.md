# Video Physics Metrics

`video_physics` group 直接比较生成视频与 simulator reference 在视频平面上的一致性。它更像是低层对齐检查：预测可以在视觉上好看，但只要颜色、位置、形状或运动区域偏离 simulator rollout，这组指标就会下降。

## 输入

| 输入 | 用途 |
| --- | --- |
| prediction video | 生成模型输出的 rollout |
| `world_gt/rgb.mp4` | simulator 渲染出的 reference RGB rollout |
| `world_gt/mask.npz`、`pred_world/mask.npz` | 物体或运动区域 mask，用于 IoU 系列指标 |

如果某些上游 evaluator 支持 fallback，它可能从 RGB 帧差分生成 motion mask；但 WorldArena 的当前 core contract 仍要求 predicted world 提供 `mask.npz`。

## 公式

设对齐后的 reference RGB 为 \(G \in [0,1]^{T \times C \times H \times W}\)，prediction RGB 为 \(P \in [0,1]^{T \times C \times H \times W}\)。实现会先截到共同帧数，并把 prediction resize 到 reference 分辨率；physics contract 还会额外检查完整 frame coverage。

### `mse`

\[
\mathrm{mse}
= \frac{1}{TCHW}
  \sum_{t=1}^{T}\sum_{c=1}^{C}\sum_{x \in \Omega}
  \left(P_{t,c,x} - G_{t,c,x}\right)^2
\]

其中 \(\Omega\) 是 \(H \times W\) 像素网格。

### IoU 系列

设对齐后的 reference mask 和 prediction mask 为：

\[
M^g_t, M^p_t \in \{0,1\}^{H \times W}, \quad t=1,\ldots,T_m
\]

基础 IoU 定义为：

\[
\mathrm{IoU}(A,B)=
\begin{cases}
\dfrac{|A \cap B|}{|A \cup B|}, & |A \cup B| > 0 \\
1, & |A \cup B| = 0
\end{cases}
\]

逐帧时空 IoU：

\[
\mathrm{st\_iou}
= \frac{1}{T_m}\sum_{t=1}^{T_m}
\mathrm{IoU}(M^g_t, M^p_t)
\]

时间 sweep 区域：

\[
S^g = \bigvee_{t=1}^{T_m} M^g_t,\qquad
S^p = \bigvee_{t=1}^{T_m} M^p_t
\]

空间 sweep IoU：

\[
\mathrm{s\_iou} = \mathrm{IoU}(S^g,S^p)
\]

weighted sweep IoU 的 reference occupancy weight 为：

\[
w_x = \frac{1}{T_m}\sum_{t=1}^{T_m} M^g_t(x)
\]

若 \(\sum_x w_x \le \epsilon\)，实现会令所有 \(w_x=1\)。记：

\[
I_x = S^g_x \land S^p_x,\qquad
U_x = S^g_x \lor S^p_x
\]

则：

\[
\mathrm{ws\_iou}
=
\begin{cases}
\dfrac{\sum_{x \in \Omega} w_x I_x}{\max\left(\sum_{x \in \Omega} w_x U_x,\epsilon\right)}, & \sum_x U_x > 0 \\
1, & \sum_x U_x = 0
\end{cases}
\]

这里 \(\epsilon=10^{-6}\)。

## 指标

| 指标 | 计算方式 | 原始分数含义 | 方向 | 当前归一化 |
| --- | --- | --- | --- | --- |
| `mse` | 对齐后的 prediction RGB 与 reference RGB 做逐像素均方误差 | 直接重建误差。它会同时惩罚颜色、纹理、形状、空间位置和时序偏差。 | 越低越好 | `[0.0, 0.25]` |
| `st_iou` | 每一帧分别计算 reference mask 与 prediction mask 的 IoU，再取平均 | spatio-temporal occupancy overlap。它检查物体或运动区域是否在每个时间步都落在正确位置。 | 越高越好 | `[0.0, 1.0]` |
| `s_iou` | reference / prediction mask 沿时间取并集后计算 IoU | swept-area overlap。它忽略具体时间，只看整个 rollout 中被占据过的空间范围是否一致。 | 越高越好 | `[0.0, 1.0]` |
| `ws_iou` | 在 swept-area IoU 上用 reference 中像素被占据的频率加权 | weighted swept-area overlap。它更重视 reference 中持续出现或频繁出现的区域，降低偶发边界像素的影响。 | 越高越好 | `[0.0, 1.0]` |

## 适合解读的问题

- 预测视频是否在像素层面接近 simulator reference。
- 物体或运动区域是否覆盖了正确空间范围。
- 运动主体是否在每一帧保持大致正确的位置。

## 不适合单独解读的问题

- `mse` 对视觉外观很敏感，不等价于物理正确性。
- `s_iou` 和 `ws_iou` 会弱化时间顺序，轨迹在错误时间经过正确位置时仍可能较高。
- `st_iou` 依赖 mask 质量；mask artifact 缺失或错位时，分数会反映 artifact 问题而不只是生成问题。

## 常见失败形态

- 预测主体位置漂移，导致 `st_iou` 下降。
- 物体经过的空间范围太大或太小，导致 `s_iou` / `ws_iou` 下降。
- 颜色、材质或背景和 simulator reference 偏差很大，导致 `mse` 上升。
- frame count 或 coverage 不满足契约时，整个 physics sample 会先触发 contract failure。
