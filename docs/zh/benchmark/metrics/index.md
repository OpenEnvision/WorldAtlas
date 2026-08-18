# 指标总览

本文按 `config/benchmark.yaml` 中的公开 benchmark 面组织。四个 formal suite 是公开 benchmark 主体；`experimental` 是补充切片，不应混成 benchmark 总分。

## 读数规则

聚合报告中的字段含义如下：

- **`raw`** — 后端原始值；可能是标量或分量字典
- **`normalized`** — 写入聚合的 `[0, 1]` 分数；统一约定越高越好
- **`official`** — 进入 `per_metric_summary.json` 和 `per_dimension_summary.json`
- **`diagnostic`** — 非 official；永远不进 headline 聚合
- **`not_applicable`** — 强监督不足，不计算、不聚合；进入 `not_applicable_summary.json`
- **`normalized = null`** — runtime 失败或缺输入；不会被自动记为 0

各指标的归一化方式、适用 suite 与 official 门槛，见对应指标页。

## Suite 向量

| Suite | 指标数 | 说明 |
| --- | ---: | --- |
| `image_static` | 43 | 以 image 为条件的静态世界生成；video prediction 在输入允许时按 video 指标面扩展评测 |
| `image_dynamic` | 33 | 同 `image_static`，通常覆盖更多 motion prompt |
| `video_static` | 37 | 以 video 为条件的静态世界生成 |
| `video_dynamic` | 44 | 以 video 为条件的动态世界生成 |
| `experimental` | 3 | 补充切片，不参与 formal headline 汇总 |

哪些指标在某个 suite 中为 official、diagnostic 或 `not_applicable`，取决于监督信号与后端是否成功；细节见各指标页与 [基准评测总览](../index.md)。

## 六个评测支柱

Metrics 导航按六个顶层维度组织；每个维度有独立索引页，汇总该维度的指标族与解读重点。

| 维度 | 分类页 |
| --- | --- |
| Long sequence（memory） | [Long Sequence（Memory）](dimensions/long_sequence.md) |
| Action control | [Action Control](dimensions/action_control.md) |
| 3D/4D consistency | [3D/4D Consistency](dimensions/consistency_3d_4d.md) |
| Physics | [Physics](dimensions/physics.md) |
| Quality | [Quality](dimensions/quality.md) |
| Real time | [Real Time](dimensions/real_time.md) |
