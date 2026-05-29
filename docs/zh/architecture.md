# 架构

WorldArena 应该坚持“**基准优先**”，而不是“pipeline 优先”。仓库需要对 manifest、模型配置、指标和报告暴露出一套一致的契约，同时允许本地仓库、API 提供方和重量级指标源码在基准核心之外独立演进。

## 模块边界

| 模块 | 职责 | 应放置的内容 |
| --- | --- | --- |
| `worldarena/benchmark/` | 评测契约 | manifest、协议、指标适配器、归一化、报告 |
| `worldarena/datasets/` | 数据集发现 | 原始资产扫描、sidecar 发现、清单生成 |
| `worldarena/models/` | 被评测模型与辅助运行时 | 本地仓库适配器、API 适配器、provider client、模型配置解析 |
| `worldarena/common/` | 共享基础设施 | taxonomy、运行时布局、序列化 |
| `config/models/` | 统一模型注册表 | 适用于本地模型与 API 模型的单一 YAML 格式 |
| `thirdparty/` | 链接的上游代码 | 模型仓库和重型指标源码 checkout，但不放被 vendoring 的 benchmark 逻辑 |

关键约束是：评测逻辑不应按模型家族分叉。无论是 3D 模型、4D 模型、API 视频模型，还是本地 world model，都应该通过同一套模型配置契约和报告契约进入系统。

## 核心契约

WorldArena 的公开接口应以数据契约为中心。内部 runner 和 adapter 可以演进，但下面这些产物决定了一次 run 是否可复现、可对比、可调试：

| 契约 | 归属 | 稳定字段 | 扩展规则 |
| --- | --- | --- | --- |
| Manifest sample | `worldarena_manifest.jsonl`，对应 `BenchmarkSample` | `sample_id`、`suite`、`split`、`reference_path`、`prediction_stem`、条件输入字段、prompt、domain tag、任务元数据、监督 sidecar、媒体 probe 字段 | 新增监督或 artifact 元数据时优先加 nullable 字段。manifest 已生成后，不要再靠目录名反推 benchmark 语义。 |
| Model runtime config | `config/models/*.yaml`，对应 `ModelRuntimeConfig` | 模型身份、`backend_type`、runtime entrypoint 或 API provider、`supported_suites`、`task_families`、`artifact_types`、`control_signals`、`roles`、`generation`、`metric_overrides` | 本地仓库、Python module、SDK API 和 HTTP API 共用一套 YAML 形状。provider 私有信息放进 `api` 或 `metadata`，不要改变顶层契约。 |
| Prediction artifact | adapter / provider 写到 `predictions/` 的输出 | `<suite>/<prediction_stem>.<ext>` 或 `<prediction_stem>.<ext>`，扩展名由 prediction modality 决定 | prediction 是被评测的主产物。额外帧、mesh、depth map 或日志应作为显式 sidecar，不应改变必需 prediction 文件名。 |
| Metric output | `per_sample_scores.jsonl` 中的 `MetricOutput` | `raw`、`normalized`、`backend`、`details`、`protocol_name`、`eligibility_status`、`error` | raw measurement 和 normalized score 分开存。用 `official`、`diagnostic`、`not_applicable` 表达聚合语义，不要把这些语义编码进指标名。 |
| Report bundle | `run_manifest.json`、summary、breakdown 和 `leaderboard.csv` | run metadata、missing prediction 数量、reported metrics、官方 summary、诊断 summary、failure summary、domain breakdown | 保持 official、diagnostic、not-applicable 和 failure 四类输出分离。除非显式版本化权重契约，否则不要新增 benchmark 级总分。 |

生命周期应保持单向：dataset scanner 生成 inventory，manifest 冻结样本语义，model adapter 按 `prediction_stem` 生成 prediction，metric 只读取 manifest 与 prediction artifact，report 只从 metric output 聚合。adapter 不应通过数据集目录结构判断资格，metric 也不应按模型家族分叉。

扩展 benchmark 时，优先复用已有轴：新任务类别放进 `task_family`，新输出类型放进 `artifact_type`，新输入条件放进 `control_signals`，辅助模型身份放进 `roles`，新评分组挂到 metric dimension。确实需要新增 enum value 时，要同步更新 taxonomy、配置默认值、测试和文档。

## 任务家族

当前代码已经在模型配置层和 manifest 层都显式携带任务元数据：

| 字段 | 示例 | 作用 |
| --- | --- | --- |
| `task_families` | `video_generation`, `scene_3d_generation`, `scene_4d_generation`, `world_model`, `reconstruction_3d`, `foundation_oracle` | 让基准扩展保持有组织，而不需要新建独立仓库 |
| `artifact_types` | `image`, `video`, `mesh`, `gaussian_splat`, `point_cloud`, `multiview`, `depth_map` | 显式声明输出及辅助产物类型 |
| `control_signals` | `text`, `image`, `video`, `camera_pose`, `actions`, `multi_view`, `depth` | 保持模型能力描述与指标资格判断一致 |
| `backend_type` | `local_repo`, `python_module`, `sdk_api`, `http_api` | 让 API 模型与本地模型共用同一 YAML 契约 |
| `roles` | `candidate`, `oracle`, `metric_backend`, `reconstruction`, `foundation` | 防止把目标模型和辅助评估模型混在一起 |

在当前正式 suite 中，样本仍然解析为 `task_family = video_generation`。偏物理模拟器风格的轨道则解析为 `task_family = world_model`。这样仓库就能扩展到 3D、4D 和 world-model 覆盖面，而不用再引入第二套 benchmark 命名空间。

## 模型接入模式

### 本地仓库

使用 `backend_type: local_repo`，并把 `repo_root` 指向 `thirdparty/` 下的仓库，或其他本地路径。

### API 模型

使用同样的 YAML 结构，但设置为：

```yaml
backend_type: http_api
repo_root: null
entrypoint: null
api:
  provider: example
  api_key_env: EXAMPLE_API_KEY
```

模板位于 `config/models/template_video_api.yaml.example`。

这就是替代“从其他仓库复制一整棵 pipeline 代码树”的推荐方案。如果 provider 需要自定义逻辑，就在 `worldarena/models/adapters/` 增加一个轻量适配器，在 `worldarena/models/providers/` 增加 provider client，但不要改变对外公开的配置格式。

## 指标与指标包

指标实现应按 benchmark 维度组织，而不是按 `basic`、`advanced` 这类主观标签组织。当前 taxonomy 是：

- `long_sequence`
- `action_control`
- `consistency_3d_4d`
- `physics`
- `quality`
- `real_time`

这套 taxonomy 是故意保持粗粒度的。六个 key 描述的是 benchmark 级能力覆盖面，而不是几个彼此割裂的小桶，也不意味着今天公开 benchmark 已经覆盖了所有可能的评测模式。

`action_control` 的定义故意比“纯 prompt following”更宽。它可以继续拆成 scene control、camera control 和 object/action control，从而把纯文本 benchmark、pose-conditioned run 以及未来的 keyboard / mouse 交互式轨道都放进同一套 taxonomy 中。

新增指标源应作为“维度化的指标包”挂接进来，而不是演变成第二套 benchmark 系统。例如：

- `Pulse-of-Motion` 应服务于 `real_time` 维度，尽管它的 runtime 配置键仍然位于 `benchmark.metric_runtime.temporal_calibration`。
- 3D 重建和多视角指标应归入 `consistency_3d_4d`。
- simulator-backed world metric 应归入 `physics`。

这样既能维持统一的报告接口，也能让评测面比当前视频核心更丰富。

## 推荐扩展顺序

1. 先保持当前正式视频 benchmark 稳定。
2. 通过统一的 YAML + adapter/provider 模式接入 API 视频模型。
3. 将 3D / 4D / reconstruction manifest 作为新的任务家族加入，而不是拆成独立仓库。
4. 将诸如 real-time calibration 或 simulator-backed physics 之类的新指标包作为可选维度叠加到同一套报告契约上。
5. 只有在契约稳定之后，再扩大公开模型目录。
