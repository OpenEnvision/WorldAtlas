# 开发

以下命令默认都在项目的 conda 环境已经激活的前提下执行。

## 数据分析

```bash
worldarena build-docs --config config/datasets.yaml
```

这个命令会扫描配置好的数据根目录，刷新 `artifacts/dataset_stats/`，并重写 `docs/generated/`。

## 基准评测

```bash
worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest --config config/benchmark.yaml
worldarena evaluate --config config/benchmark.yaml --predictions-root /path/to/preds
```

WorldArena 的正式 benchmark 面故意保持收敛：

- 正式 suite：`image_static`、`image_dynamic`、`video_static`、`video_dynamic`
- 正式 headline 输出：按指标的向量结果，而不是 suite 单分，也不是全局总分
- 缺少强监督的指标会标为 `not_applicable`，不会混入正式或诊断报告

simulator-backed physics 工作流使用独立配置：

```bash
worldarena build-physics-manifest --physics-pipeline-root /path/to/physics-pipeline
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --physics-pipeline-root /path/to/physics-pipeline
```

## 集成模型适配器

内置适配器假设 `config/models/*.yaml` 使用对开源友好的相对路径。如果某个上游仓库需要不同布局，优先在本地覆盖 YAML，或把仓库链接进 `thirdparty/`，而不是把私有绝对路径硬编码回版本化配置中。

所有模型，包括未来的 API 模型，都应保持同一套 YAML 表面：

- `backend_type`
- `task_families`
- `artifact_types`
- `control_signals`
- `roles`

对于 API 模型，请把 `config/models/template_video_api.yaml.example` 作为起点，而不是另外发明第二种配置格式。

当前 API 适配器走的是通用 benchmark 路径，而不是某个服务专用的旁路框架：

- `backend_type: http_api`
- 同一个 request builder 同时支持 `t2v` 和 `i2v`
- provider 特定的提交与轮询逻辑位于 `model.api`
- benchmark 命令路径不为远程模型做特殊分叉

`config/models/kling_api.yaml.example` 是具体参考配置。API pipeline 的回归测试跑在本地 fake polling server 上，因此无需真实凭证也能验证 benchmark 契约。

## 预览与校验

```bash
mkdocs build --strict
pytest
ruff check .
```

如果只想做一轮更轻量的 pre-commit 校验，至少运行：

```bash
pytest -q tests/test_model_config.py tests/test_runtime_layout.py tests/test_environment_check.py
```

如果你修改了重型运行时 wrapper 或 simulator-backed evaluation，建议再补跑对应的定向测试，例如 `tests/test_met3r_runtime.py`、`tests/test_long_sequence_runtime.py`、`tests/test_long_sequence_native.py` 和 `tests/test_physics_benchmark.py`。

## 清理

```bash
worldarena clean
```

如果还想删除 `docs/generated/` 和 `artifacts/dataset_stats/`，请加上 `--generated`。

## 开源发布卫生

在发布改动之前，检查以下事项：

- `config/models/*.yaml` 应使用项目相对默认值，而不是私有绝对路径。
- `cache/` 和 `thirdparty/` 在版本控制中通常只应保留被跟踪的 README 文件。
- 文档应说明新增适配器所带来的 checkpoint、仓库或许可证假设。
- 生成出的 benchmark 报告和数据分析产物一般不应提交进 git，除非它们是有意纳入版本控制的 fixture。
