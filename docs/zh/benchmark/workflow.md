# 工作流与输出

以下命令默认都在项目的 conda 环境已经激活的前提下执行。

## 主 Benchmark 工作流

`build-manifest` 已经会刷新 benchmark inventory 产物，所以标准最短路径是：

```bash
worldarena build-manifest --config config/benchmark.yaml
worldarena generate --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
worldarena evaluate --config config/benchmark.yaml --predictions-root artifacts/predictions/lingbot_world
```

一条命令跑完的路径：

```bash
worldarena run-model --config config/benchmark.yaml --model-config config/models/lingbot_world.yaml
```

如果你只是想单独查看原始资产审计结果，而不立刻写 manifest，可以单独运行：

```bash
worldarena build-inventory --config config/benchmark.yaml
```

## 常用过滤项与覆盖项

- `--suites image_static,video_dynamic`
- `--limit 32`
- `--overwrite` 用于重新生成
- `--manifest /path/to/worldarena_manifest.jsonl` 用于固定某个 manifest snapshot
- 在 `evaluate` 上加 `--model-config ...`，用于先应用模型特定的 metric override 再开始打分

## 预测目录布局

接受的目录布局包括：

```text
predictions/
  image_static/<sample_id>.mp4
  image_dynamic/<sample_id>.mp4
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

evaluator 也会解析：

- `predictions/<suite>/<sample_id>.<ext>`
- `predictions/<sample_id>.<ext>`

支持的文件类型由 benchmark 预测契约决定。当前 formal suite 都以 video prediction 评测；`image_static` / `image_dynamic` 是 image-conditioned video rollout，不是静态图片输出。

支持的文件类型：

- video：`.mp4`、`.mov`、`.avi`、`.mkv`、`.webm`
- 非 formal 或后续 image artifact：`.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`

## 输出文件

| 文件 | 作用 | 你应先看它的场景 |
| --- | --- | --- |
| `run_manifest.json` | 运行元数据、协议、`missing_predictions`、`reported_metrics`、`reported_diagnostics` | 你想先确认 run contract 是否正确 |
| `per_metric_summary.json` | 正式 suite-by-metric 平均值 | 你想看 headline 结果 |
| `per_dimension_summary.json` | 基于六个顶层 benchmark 维度计算的正式 suite-by-dimension 平均值 | 你想先看更紧凑的维度汇总 |
| `diagnostic_metric_summary.json` | 诊断指标的平均值 | 你想看额外但非 headline 的分析信号 |
| `not_applicable_summary.json` | 因强监督不足而跳过的指标计数和原因 | 你想确认哪些指标没有资格打分 |
| `per_sample_scores.jsonl` | 每个样本的 `raw`、`normalized`、`backend`、`eligibility_status`、`details`、`error` | 你需要调试或做消融 |
| `domain_breakdown.json` | 按 benchmark domain 分组的官方指标 | 你需要结构化 failure slicing |
| `leaderboard.csv` | 官方 suite 指标的扁平导出 | 你想导入表格工具 |

只有在 `missing_predictions = 0`，并且使用了同样的 suite、manifest snapshot 和 metric config 时，评测输出才可直接比较。

## 正式对比规则

如果你要对多个模型做 leaderboard 式正式对比：

- 只构建一份 manifest snapshot，并对所有模型复用
- 所有可比 run 都应保持相同的 suite、manifest snapshot 和 metric 配置

## Simulator-Backed Physics 工作流

simulator-backed physics track 使用独立的配置和 manifest 根目录。完整契约现在并入 Metrics 下的 [Physics](metrics/dimensions/physics.md)、[Simulator-Backed Physics](metrics/physics/index.md)、[Physics Workflow](metrics/physics/workflow.md) 和 [Physics Metrics & Contracts](metrics/physics/metrics.md)；这里先给出最短操作路径：

从 physics pipeline 导出结果构建 manifest：

```bash
worldarena build-physics-manifest \
  --physics-pipeline-root /path/to/physics-pipeline
```

针对该 manifest 评测预测结果：

```bash
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

为 i2v continuation case 构建 image-seed physics manifest：

```bash
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --physics-pipeline-root /path/to/physics-pipeline
```

这些命令复用同一套 report contract，但它们面向的是 simulator-backed `experimental` suite，而不是四个 headline suite。

几个实际要点：

- 如果不传 `--output-dir`，默认报告目录是 `artifacts/benchmark/physics/reports/<model_name>`
- prediction 路径解析与主 benchmark 相同，因此文件既可以放在 `experimental/`，也可以直接放在预测根目录
- `per_sample_scores.jsonl` 中会带上 `physics_status`、`world_gt_status`、`pred_world_status`、`metric_contract` 和 `missing_world_artifacts`
