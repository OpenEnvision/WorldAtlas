# 开发

<p class="dimension-lede">本页是 WorldAtlas Arena 的贡献者工作流索引：环境入口、数据刷新、benchmark 命令、文档构建与校验。模块边界与契约见 <a href="architecture/">架构</a>；完整运行时安装见 <a href="thirdparty/">运行时布局</a>。</p>

以下命令默认在 Conda 环境（`worldarena`）已激活、且以 editable 方式安装本包的前提下执行：

```bash
conda activate worldarena
pip install -e .
```

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `worldarena/` | Benchmark 核心：manifest、指标、报告、模型适配器 |
| `config/` | Benchmark、数据集与模型 YAML 契约 |
| `docs/en/`、`docs/zh/` | 文档源文件（Markdown） |
| `docs/site-nav.yaml` | 文档站侧边栏与 hidden page 契约 |
| `docs-site/` | Astro v6 + Starlight 构建器；输出到 `site/doc/` |
| `site/` | 静态站点壳层、Leaderboard 与 `/doc/` 托管 |
| `thirdparty/` | 链接的上游模型与指标运行时 |
| `artifacts/` | 预测、报告与生成的数据统计 |

## 数据分析

刷新数据集 inventory 面板与 generated 分析产物：

```bash
worldarena build-docs --config config/datasets.yaml
```

该命令扫描配置的数据根目录，更新 `artifacts/dataset_stats/`，并重写 `docs/generated/`。新增、删除或重组源数据后应重新运行。分析页上的面板反映最近一次成功扫描的结果。

## 文档站点

文档由 **Astro + Starlight** 构建（不再使用 MkDocs）。源文件位于 `docs/{en,zh}/`；导航由 `docs/site-nav.yaml` 驱动。

**安装文档工具链**（Node.js 18+）：

```bash
cd docs-site
npm install
```

**本地预览**（热更新，`http://localhost:4321/doc/`）：

```bash
npm run dev
```

**生产构建**（输出静态 HTML 到 `site/doc/`）：

```bash
# 在仓库根目录
npm run docs:build
```

构建流程会自动：

1. 将 `docs/{en,zh}` 同步到 `docs-site/src/content/docs/`（连字符 slug、资源路径改写）。
2. 重新生成 `docs-site/src/sidebar.generated.mjs`。
3. 将 `docs/assets/` 中的共享资源链接到 docs public 目录。

**与主站一起预览**（`/doc/` 与 Leaderboard、Arena 同服）：

```bash
npm run docs:build
npm run site
# 默认 http://localhost:4173/doc/
```

旧路径 `/docs/*` 会 301 重定向到 `/doc/*`。编辑带 case 媒体的 metric 页时，需保证 `docs/assets/metric_examples/` 存在；同步步骤会将其链接到 `/assets/worldarena/assets/metric_examples/`。

更多路由映射与样式定制见 `docs-site/README.md`（`docs-site/src/styles/custom.css`）。

## 基准评测

Formal suite 标准流程：

```bash
worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest --config config/benchmark.yaml

# 用已注册 adapter 生成预测
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml

# 评估已有 predictions 目录
worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root artifacts/predictions/<model-run>

# 或一步 generate + evaluate
worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

WorldAtlas Arena 的公开 benchmark 面刻意保持收敛：

- **Formal suite：** `image_static`、`image_dynamic`、`video_static`、`video_dynamic`
- **Headline 输出：** `per_metric_summary.json` 中的按指标向量，以及 `per_dimension_summary.json` 中的维度汇总——不是 suite 单分，也不是全局总分
- **适用性：** 缺少强监督的指标标为 `not_applicable`，不进入 official 聚合

Simulator-backed physics 使用独立配置与 report namespace：

```bash
worldarena build-physics-manifest --physics-pipeline-root /path/to/physics-pipeline
worldarena build-physics-seed-manifest \
  --image-manifest /path/to/seed-image-manifest.jsonl \
  --physics-pipeline-root /path/to/physics-pipeline
```

解释边界见 [Benchmark 总览](benchmark/index.md) 与 [Physics 维度](benchmark/metrics/dimensions/physics.md)。

## 集成模型适配器

内置适配器假设 `config/models/*.yaml` 使用对开源友好的相对路径。若上游仓库布局不同，优先在本地覆盖 YAML，或将仓库链接到 `thirdparty/`，不要把私有绝对路径写回版本化配置。

所有模型——本地 repo、Python module、HTTP API——共用同一套 YAML 表面：

| 字段 | 含义 |
| --- | --- |
| `backend_type` | `local_repo`、`python_module`、`sdk_api` 或 `http_api` |
| `task_families` | 如 `video_generation`、`world_model` |
| `artifact_types` | 输出模态（`video`、`mesh` 等） |
| `control_signals` | 用于 eligibility 的条件轴 |
| `roles` | `candidate`、`oracle`、`metric_backend` 等 |

API 模型请以 `config/models/template_video_api.yaml.example` 为起点，不要另起一套配置格式。通用路径为：

- `backend_type: http_api`
- 同一 request builder 支持 `t2v` 与 `i2v`
- provider 提交/轮询逻辑位于 `model.api`
- benchmark CLI 不为远程模型单独分叉

`config/models/kling_api.yaml.example` 是具体参考。API pipeline 测试跑在本地 fake polling server 上，无需真实凭证即可验证契约。

## 预览与校验

**文档**

```bash
npm run docs:build
```

**Python**

```bash
pytest
ruff check .
```

修改模型配置或 runtime layout 时，至少跑：

```bash
pytest -q tests/test_model_config.py tests/test_runtime_layout.py tests/test_environment_check.py
```

若改动重型 runtime wrapper 或 simulator-backed evaluation，建议补跑：

- `tests/test_long_sequence_backend_runtime.py`
- `tests/test_long_sequence_runtime.py`
- `tests/test_long_sequence_native.py`
- `tests/test_physics_simulator_derived_metrics.py`

编辑 `docs/` 下导航或 Markdown 后，请执行 `npm run docs:build` 并在主站 `/doc/` 下抽查。

## 站点内容导出

刷新静态站点壳层使用的 SPA 内容包（`site/js/content*.js`）：

```bash
worldarena build-site-content
```

该命令读取 `docs/site-nav.yaml` 并导出页面正文。`/doc/` 下的 Starlight 页面由 `npm run docs:build` 单独构建。

## 清理

```bash
worldarena clean
```

加上 `--generated` 可同时删除 `docs/generated/` 与 `artifacts/dataset_stats/`。
