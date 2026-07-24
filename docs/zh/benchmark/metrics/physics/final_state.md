# Final State Accuracy

**final_state_accuracy** 是面向 final-state classification 任务的场景条件 simulator-backed 指标，例如 lever 最终状态。它判断 prediction 的最终状态是否与 ground truth 最终状态一致。

该指标已经配置在 `config/benchmark_physics.yaml` 中，但只有样本在 `physics_spec.final_state_metric` 中声明 final-state 规则或显式标签时才会计算；否则会记为 `not_applicable`。

## 输入

| 输入 | 用途 |
| --- | --- |
| `physics_spec.final_state_metric` | 声明显式标签，或声明如何从 simulator worldline 推导标签 |
| `world_gt/worldlines.npz` | 未提供 `true_label` 时，用来推导 ground-truth label |
| `pred_world/worldlines.npz` | 未提供 `predicted_label` 时，用来推导 prediction label |

## 支持的契约

可以直接提供显式标签：

```yaml
physics_spec:
  final_state_metric:
    true_label: left_down
    predicted_label: left_down
```

对于 lever 类场景，可以用两个端点 track 推导标签：

```yaml
physics_spec:
  final_state_metric:
    type: lever
    left_track_index: 0
    right_track_index: 1
    axis: y
    balanced_tolerance: 0.02
    lower_is_down: true
```

默认标签为 `left_down`、`right_down` 和 `balanced`。可以通过下面字段覆盖：

```yaml
labels:
  left_down: left_down
  right_down: right_down
  balanced: balanced
```

## 公式

对单个样本：

\[
\mathrm{final\_state\_accuracy}
=
\mathbf{1}\left[\hat{y}=y\right]
\]

对 \(S\) 个样本聚合时取平均：

\[
\mathrm{Accuracy}
=
\frac{1}{S}\sum_{i=1}^{S}
\mathbf{1}\left[\hat{y}^{(i)}=y^{(i)}\right]
\]

## Lever 规则

对 lever rule，WorldArena 会读取两个端点都有效的最后一帧。若 `axis: y` 且 `lower_is_down: true`：

- 左端点比右端点低超过 `balanced_tolerance` 时为 `left_down`
- 右端点比左端点低超过 `balanced_tolerance` 时为 `right_down`
- 否则为 `balanced`

如果没有提供显式标签，reference 和 prediction worldline 都用同一规则推导标签。

## 输出

| 字段 | 含义 |
| --- | --- |
| Raw score | final state 一致时为 `1.0`，否则为 `0.0` |
| Normalized score | `[0, 1]` 上的恒等映射 |
| Eligibility | 样本提供 final-state contract 时为 official；否则为 `not_applicable` |

## 失败模式

- 样本声明了 final-state 指标，但既没有显式标签，也没有支持的推导规则。
- lever track index 缺失或超出 worldline 数组范围。
- 选中的 final-state track 没有共同有效帧。
