# Physics Workflow

以下命令默认都在项目的 conda 环境已经激活的前提下执行。

## 必需的外部 Checkout

simulator-backed 路径依赖一个通过 `--physics-pipeline-root` 显式传入的外部 physics-pipeline checkout。

| 路径或文件 | 作用 |
| --- | --- |
| `assets/<sample_id>.mp4` | 参考 rollout；也用于导出默认 conditioning frame |
| `assets/<sample_id>.json` | prompt 元数据、物体 taxonomy、scene tag 和 simulator hint |
| `benchmark_export.py` | 把 `world_gt` artifact 导出到 physics data root |
| `benchmark_eval.py` | 基于 `world_gt`、生成视频和可选 `pred_world` artifact 计算 simulator-backed 指标 |
| `main_part2.py` | 当当前 metric contract 需要 `pred_world` 时，为生成视频重建 `pred_world` artifact |

当前的 core contract 默认要求这三个脚本都存在。缺失 `benchmark_export.py` 或 `benchmark_eval.py` 会让 case 进入 `missing_backend`；缺失 `main_part2.py` 通常会在之后表现为 `pred_world` contract failure。

## 路径解析

- `build-physics-manifest` 和 `build-physics-seed-manifest` 会把显式传入的 `physics_pipeline_root` 记录到每个样本的 `physics_case` 中。
- physics data root 默认解析为 `<physics_pipeline_root.parent.parent>/.runtime/physics_pipeline/data_root`。
- 在不传 `--output-dir` 时，benchmark runner 会把报告写到 `artifacts/benchmark/physics/reports/<model_name>`。

physics run 的 prediction 路径解析沿用普通 benchmark 的规则，因此下面几种布局都可接受：

- `predictions/experimental/<sample_id>.mp4`
- `predictions/<sample_id>.mp4`

## 从 Simulator 资产构建 Manifest

标准批量路径：

```bash
worldarena build-physics-manifest \
  --physics-pipeline-root /path/to/physics-pipeline
```

默认输出根目录是 `artifacts/benchmark/physics/manifest`，其结构如下：

```text
artifacts/benchmark/physics/manifest/
  conditioning/<source_sample_id>.png
  cases/physics__<source_sample_id>__i2v/
    simulator/
      metadata_template.json
      submitted_video.mp4
    world_gt/
  worldarena_physics_manifest.jsonl
  worldarena_physics_manifest_summary.json
```

注意：

- `conditioning/<source_sample_id>.png` 会在缺失时从参考 rollout 的首帧导出
- summary JSON 会记录样本数以及当前 `required_core_metrics`
- 样本 id 是确定性的：`physics__<source_sample_id>__i2v`

## 评测预测结果

```bash
worldarena evaluate \
  --config config/benchmark_physics.yaml \
  --manifest artifacts/benchmark/physics/manifest/worldarena_physics_manifest.jsonl \
  --predictions-root /path/to/physics-preds
```

每个样本的评测生命周期如下：

1. 先把提交的视频 stage 到 case 本地的 `simulator/submitted_video.mp4`。
2. 调用 `inspect_physics_case(...)` 检查当前就绪状态。
3. 如果本地 `world_gt` 不完整，就尝试从 physics pipeline 导出或同步已存在的导出副本。
4. 如果本地参考视频仍然不可用，则直接失败，并返回 `physics_status`、remediation 提示和缺失 artifact 名称。
5. 如果允许自动生成 `pred_world`，就运行 `main_part2.py` 并导出预测世界状态 artifact。
6. 运行 `benchmark_eval.py`，写出 `<prediction_stem>_physics_report.json`。
7. 把 `physics_status`、`world_gt_status`、`pred_world_status`、`metric_contract`、`missing_world_artifacts` 等诊断字段附加到 benchmark 输出里。

## Readiness 状态

`inspect_physics_case(...)` 在自动补救前后会报告以下状态之一：

| 状态 | 含义 |
| --- | --- |
| `ready` | 本地 `world_gt` 已经具备完整的必需 metric artifact 集 |
| `needs_sync` | physics data root 下已经存在完整导出的 `world_gt`，可以同步到本地 case |
| `missing_backend` | 缺少 `benchmark_export.py` 或 `benchmark_eval.py` 等必需 pipeline 脚本 |
| `partial_world_gt` | 本地 `world_gt` 至少已有 `rgb.mp4`，但完整 metric artifact 仍然不全 |
| `needs_export` | evaluator 必须先导出 `world_gt`，case 才能满足 metric contract |

`preflight_physics_benchmark(...)` 会对一批样本执行同样的检查，并记录缺失 metric artifact 与 remediation 命令。

## Seed Image Manifest 流程

seed-image 路径会把生成好的种子图像转换成 simulator-backed continuation case：

```bash
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --catalog /path/to/physics-seed-catalog.json \
  --physics-pipeline-root /path/to/physics-pipeline
```

选择规则：

- 每条输入记录都必须有非空 `case_id`
- 只有 `status = ok` 的记录才会被保留
- 还可以通过 `--case-ids` 进一步过滤
- 如果传了 `--catalog`，会先按 `case_id` 合并 catalog，再由 image-manifest 记录覆盖冲突键

最终生成的样本 id 为 `physics_seed__<case_id>__i2v`。builder 还会把 `motion_family`、`interaction_family`、`difficulty_tier`、`support_geometry`、物体元数据和 simulator override 等结构化字段一路带进 `physics_spec` / `physics_case`。

## 仅 Python 的在线路径

做单样本调试时，可以跳过 batch manifest CLI，直接在 Python 里评测一个 i2v case：

```python
from worldarena.benchmark.physics_backend import evaluate_online_physics_i2v

result = evaluate_online_physics_i2v(
    prompt="A toy ball rolls down a wooden incline.",
    image_path="tmp/seed.png",
    generated_video_path="tmp/pred.mp4",
    output_root="tmp/physics_online",
    physics_pipeline_root="/path/to/physics-pipeline",
)
print(result["success"], result["report_path"])
```

这条路径会构建一个 `online_physics__...` 样本，运行外部 simulator pipeline，导出 `pred_world`，并返回与 batch evaluator 相同风格的结果负载。
