# WorldModel Benchmark Detailed Spec

## 1. 目标与范围

这份文档不是综述，而是给后续 coding agent 的**实现规格书**。目标是把你已经按 WorldScore 分类方式收集好的高质量 image / video 数据，组织成一个可执行的 benchmark，使后续 agent 可以直接按本文实现：

1. 数据 schema
2. 数据获取与许可分层
3. 原始数据到 benchmark sample 的加工流程
4. 统一预处理
5. 每个 metric 的输入 / 输出
6. 使用的模型与 checkpoint
7. 计算公式
8. 逐步算法流程
9. score normalization
10. 聚合与报表

默认 benchmark 同时支持两类评测：

- **reference-grounded**：有参考视频、相机轨迹、动作序列或 reference state，可做精确对齐评测
- **physics-grounded**：没有唯一 ground truth，但有明确物理规律、守恒量或 counterfactual 方向，可做 rule-based / invariant-based 评测

本文默认使用 WorldScore 的三大主轴作为总框架，再补齐长时建模、counterfactual、physics、interactive 四块：

- **Controllability**
- **Consistency / Quality**
- **Dynamics / Physics**
- **Interactive**

## 2. Benchmark Track 设计

建议把整个 benchmark 拆成 8 个 track，每个 sample 只属于一个主 track，但可以共享部分 metric。

| Track ID | 名称 | 主要目标 | 是否需要 reference |
| --- | --- | --- | --- |
| `long_revisit` | Long-range static revisit consistency | 长时间绕行后能否回到同一布局 | 是 |
| `long_delayed_effect` | Long-range delayed-effect prediction | 早期事件在晚期是否正确显现 | 最好有 |
| `camera_control` | Camera controllability | 是否按照文本 / action / trajectory 控制镜头 | 是 |
| `object_control` | Object controllability | 是否按控制信号生成 / 保持 / 操作目标对象 | 是 |
| `scene_control` | Scene / content alignment | 是否生成目标 scene content 并保持语义对齐 | 是 |
| `geom_consistency` | 3D / 4D consistency | 多视角 / 时序几何一致性 | 否 |
| `physics` | Basic physics + physical invariance + causal consequence | 是否符合动力学与物理规律 | 否，部分子项可有 |
| `interactive` | Real-time interactive world modeling | 在线响应延迟、持续控制稳定性 | 是 |

说明：

- 如果某个 sample 有 reference video，就优先走 reference-grounded metric。
- 如果某个 sample 只有“物理现象成立”这一类弱监督，就走 physics-grounded metric。
- 对一个模型的最终结果，长期 taxonomy 可以保留：
  - `WorldModelScore-Core`
  - `WorldModelScore-Interactive`
- 但**当前 WorldArena benchmark 的实际输出名**应以：
  - `S_image_static`
  - `S_image_dynamic`
  - `S_video_static`
  - `S_video_dynamic`
  - `S_worldarena`
  - `S_auxiliary`
  - `S_experimental`
  为准。

对于**当前 WorldArena benchmark**，上表只是完整 taxonomy，不是一次性全部上线的承诺。默认分为三层：

- `official`：当前数据已经能完整闭环支持的 track，会进入主 leaderboard。
- `experimental`：当前数据可以做弱版或代理版评测，但不进入主总分。
- `future-expansion`：必须等后续补数据才能做的 track。

基于现有数据，默认划分如下：

- `official`：四个 formal suite，即 `image_static`、`image_dynamic`、`video_static`、`video_dynamic`
- `experimental`：subset-level `camera_control/object_control/geom`、`long_delayed_effect`、`basic physics`
- `future-expansion`：`long_revisit`、完整 `interactive`、更强 supervision 的 control / geometry / physics 主榜

当前 benchmark 推荐统一使用下列分数名：

- `S_image_static`
- `S_image_dynamic`
- `S_video_static`
- `S_video_dynamic`
- `S_worldarena`
- `S_auxiliary`
- `S_experimental`

## 3. 数据与目录 Schema

### 3.1 样本级目录结构

每个样本建议采用如下结构：

```text
data/benchmarks/worldmodel/
  sample_id/
    sample.json
    inputs/
      conditioning.png
      conditioning.mp4
      action.json
      trajectory.json
    references/
      target.mp4
      target_frames/
      camera_gt.json
      masks/
      roi/
    annotations/
      objects.json
      events.json
      physics.json
      domain_tags.json
```

### 3.2 `sample.json`

建议最小 schema：

```json
{
  "sample_id": "formal_dynamic_video_photorealistic_articulated_0001",
  "track": "scene_control",
  "asset_level": "video_dynamic",
  "is_formal": true,
  "task_type": "i2v",
  "conditioning_type": "image",
  "fps": 24,
  "num_frames": 96,
  "resolution": [1024, 576],
  "category_path": "dynamic/photorealistic/videos/articulated",
  "worldscore_tags": {
    "scene_domain": null,
    "dynamic": true,
    "style": "photorealistic",
    "pov": null
  },
  "source": {
    "source_name": "worldarena_formal_video",
    "source_type": "curated_video",
    "upstream_uid": "dynamic_photorealistic_articulated_0001",
    "license_bucket": "C",
    "redistributable": false,
    "source_group_id": "worldarena_dynamic_photorealistic_articulated_0001"
  },
  "supervision_flags": {
    "category_label": true,
    "style_label": true,
    "mask": false,
    "track": false,
    "camera_pose": false,
    "instruction": false
  },
  "inputs": {
    "conditioning_image": "inputs/conditioning.png",
    "conditioning_video": null,
    "text_prompt_current": "A photorealistic articulated-motion video.",
    "text_prompt_target": "Generate a photorealistic video with articulated object motion and consistent scene appearance.",
    "camera_instruction_text": null,
    "action_file": null,
    "trajectory_file": null
  },
  "references": {
    "reference_video": "references/target.mp4",
    "camera_gt": null
  },
  "annotations": {
    "objects": null,
    "events": null,
    "physics": null,
    "domain_tags": "annotations/domain_tags.json"
  }
}
```

补充要求：

- `asset_level` 用来描述样本的主资产类型，当前 benchmark 只允许五个值：`image_static`、`image_dynamic`、`video_static`、`video_dynamic`、`video_nonformal`。
- `is_formal` 直接决定该样本是否有资格进入 official。默认规则是：`is_formal = true` 才能进入 official；`is_formal = false` 只能进入 auxiliary / experimental。
- `category_path` 必须保留原始目录语义，例如 `static/photorealistic/indoor/living_spaces` 或 `dynamic/stylized/videos/fluid`。
- `supervision_flags` 只记录**当前数据实际具备**的监督，不允许因为未来可能补传而先写成 `true`。
- `source_group_id` 是**防泄漏主键**。来自同一上游视频、同一 capture、同一 log、同一 3D scene、同一 simulator seed family 的样本，必须共享同一个 `source_group_id`。
- `license_bucket` 不允许缺失。没有 license / terms 结论的样本，一律不允许进入冻结测试集。
- `source_type` 建议允许四类：`curated_image`、`curated_video`、`game_world`、`external_future`。

### 3.3 关键 annotation 文件

#### `objects.json`

```json
{
  "expected_objects": [
    {"name": "bed", "required": true},
    {"name": "lamp", "required": true},
    {"name": "window", "required": true}
  ],
  "controllable_objects": [
    {
      "name": "ball",
      "init_mask": "references/masks/ball_000.png",
      "expected_count": 1
    }
  ]
}
```

#### `events.json`

用于 delayed effect / action consequence。

```json
{
  "events": [
    {
      "name": "collision",
      "type": "impact",
      "reference_frame": 57,
      "tolerance": 4
    }
  ]
}
```

#### `physics.json`

```json
{
  "motion_family": "projectile",
  "primary_object": "ball",
  "gravity_axis": "y_down",
  "requires_scale_normalization": true,
  "known_invariants": ["energy", "horizontal_momentum", "vertical_acceleration"],
  "counterfactual_group": null
}
```

#### `domain_tags.json`

```json
{
  "scene_domain": "indoor",
  "weather": "clear",
  "time_of_day": "day",
  "region": "EU",
  "content_domain": "human",
  "style_domain": "photorealistic",
  "pov": "first_person"
}
```

### 3.4 规划原则：以现有 WorldArena 数据为中心

这里采用一个更务实的主张：**benchmark 不再以“未来可能补充的数据”作为前提，而是完全以当前 WorldArena 的真实数据分布表为基准来设计。**

用户已经明确说明：

- 当前 HF 只上传了部分样本，**不能**用 HF 树结构反推真实全集。
- 真实的 source of truth 是你给出的 `category,count,is_formal` 分布表。
- 当前最可靠的结构信息是：
  - `image` 与 `video`
  - `static` 与 `dynamic`
  - `photorealistic` 与 `stylized`
  - `is_formal = True / False`

因此，benchmark 必须遵守下面 8 条规则：

1. **official 只使用 `is_formal = True` 的资产。**
2. **`is_formal = False` 的资产只进入 auxiliary / experimental，不进入 official hidden test。**
3. **主榜优先依赖“明确存在的 RGB 资产 + 类别标签”，不默认假设 pose、instruction、mask、physics 标注已经齐全。**
4. **任何强监督 sidecar（mask / track / pose / instruction）只有在 inventory builder 扫描确认存在后，才能启用对应指标。**
5. **代码和文档都不允许写死最终样本数量。** 下面的数量是当前规划基线，freeze 时以 manifest 为准。
6. **static / dynamic、image / video 四大 formal suite 必须分开构建，再统一聚合。**
7. **原始 `category_path` 必须完整保留到 sample schema 与报表里。**
8. **当前 HF 镜像默认按 gated dataset 处理。** 因此 `license_bucket` 默认应设为 `C`，除非后续有内部镜像和明确的发布结论。

这一版文档后续提到的 “benchmark” 默认都指：

- **以 formal image/video 四大 suite 为核心的 benchmark**
- **以 non-formal 数据作为 auxiliary / experimental 的 benchmark**
- **以当前真实分布表作为规划基线，而不是 HF 部分上传内容**

### 3.5 现有 WorldArena 真实数据分布

本节以你给出的真实分布表为准。

#### 3.5.1 汇总视图

| Suite | Formal? | Count |
| --- | --- | ---: |
| `image/static/photorealistic` | True | 4746 |
| `image/static/stylized` | True | 2294 |
| `image/dynamic/photorealistic` | True | 1249 |
| `image/dynamic/stylized` | True | 1229 |
| `video/static/photorealistic` | True | 5999 |
| `video/static/stylized` | True | 2992 |
| `video/dynamic/photorealistic` | True | 884 |
| `video/dynamic/stylized` | True | 87 |
| `image` backups / pre-curation folders | False | 172 |
| `AAA_Games/videos` | False | 500 |

当前规划基线总量：

- `formal_image_total = 9518`
- `formal_video_total = 9962`
- `formal_total = 19480`
- `non_formal_total = 672`

#### 3.5.2 Formal image 分布

```csv
category,count,is_formal
dynamic/photorealistic/images/articulated,420,True
dynamic/photorealistic/images/deformable,257,True
dynamic/photorealistic/images/fluid,260,True
dynamic/photorealistic/images/multi_motion,203,True
dynamic/photorealistic/images/rigid,109,True
dynamic/stylized/images/articulated,255,True
dynamic/stylized/images/deformable,236,True
dynamic/stylized/images/fluid,251,True
dynamic/stylized/images/multi_motion,246,True
dynamic/stylized/images/rigid,241,True
static/photorealistic/indoor/dining_spaces,671,True
static/photorealistic/indoor/living_spaces,1051,True
static/photorealistic/indoor/passageways,201,True
static/photorealistic/indoor/public_spaces,401,True
static/photorealistic/indoor/workspaces,349,True
static/photorealistic/outdoor/aquatic_landscape,271,True
static/photorealistic/outdoor/city,526,True
static/photorealistic/outdoor/suburb,166,True
static/photorealistic/outdoor/terrestrial_landscape,694,True
static/photorealistic/outdoor/verdant_landscape,416,True
static/stylized/indoor/dining_spaces,241,True
static/stylized/indoor/living_spaces,271,True
static/stylized/indoor/passageways,206,True
static/stylized/indoor/public_spaces,241,True
static/stylized/indoor/workspaces,231,True
static/stylized/outdoor/aquatic_landscape,241,True
static/stylized/outdoor/city,200,True
static/stylized/outdoor/suburb,191,True
static/stylized/outdoor/terrestrial_landscape,261,True
static/stylized/outdoor/verdant_landscape,211,True
```

#### 3.5.3 Non-formal image 分布

```csv
category,count,is_formal
dynamic/photorealistic/_backup_before_curated_strict_v6_20260409T151731Z/images/deformable,14,False
dynamic/photorealistic/_backup_before_curated_strict_v6_20260409T151731Z/images/multi_motion,10,False
dynamic/photorealistic/_backup_before_curated_strict_v6_20260409T151731Z/images/rigid,28,False
dynamic/photorealistic/_backup_before_multi_motion_semantic_v1_20260410T053839Z/images/multi_motion,120,False
```

#### 3.5.4 Formal video 分布

```csv
category,count,is_formal
dynamic/photorealistic/videos/articulated,400,True
dynamic/photorealistic/videos/deformable,76,True
dynamic/photorealistic/videos/fluid,190,True
dynamic/photorealistic/videos/multi_motion,52,True
dynamic/photorealistic/videos/rigid,166,True
dynamic/stylized/videos/articulated,48,True
dynamic/stylized/videos/deformable,6,True
dynamic/stylized/videos/fluid,8,True
dynamic/stylized/videos/multi_motion,1,True
dynamic/stylized/videos/rigid,24,True
static/photorealistic/indoor/dining_spaces,611,True
static/photorealistic/indoor/living_spaces,616,True
static/photorealistic/indoor/passageways,500,True
static/photorealistic/indoor/workspaces,594,True
static/photorealistic/outdoor/aquatic_landscape,606,True
static/photorealistic/outdoor/city,519,True
static/photorealistic/outdoor/public_spaces,549,True
static/photorealistic/outdoor/suburb,616,True
static/photorealistic/outdoor/terrestrial_landscape,633,True
static/photorealistic/outdoor/verdant_landscape,755,True
static/stylized/indoor/dining_spaces,355,True
static/stylized/indoor/living_spaces,380,True
static/stylized/indoor/passageways,265,True
static/stylized/indoor/public_spaces,3,True
static/stylized/indoor/workspaces,375,True
static/stylized/outdoor/aquatic_landscape,350,True
static/stylized/outdoor/city,294,True
static/stylized/outdoor/public_spaces,179,True
static/stylized/outdoor/suburb,240,True
static/stylized/outdoor/terrestrial_landscape,316,True
static/stylized/outdoor/verdant_landscape,235,True
```

#### 3.5.5 Non-formal video 分布

```csv
category,count,is_formal
AAA_Games/videos,500,False
```

### 3.6 现有 WorldArena 资产分级与 supervision 假设

不要再按“外部来源”来思考，而是按**formal status + modality + dynamics + supervision** 来思考。

当前 benchmark 建议定义五类资产等级：

| `asset_level` | 当前路径模式 | 典型可用信号 | benchmark 用途 | 是否允许进入主 hidden test |
| --- | --- | --- | --- | --- |
- `image_static` | `static/...` images, `is_formal=True` | RGB image + category path + style + indoor/outdoor | official 静态图像套件 | 是 |
- `image_dynamic` | `dynamic/...` images, `is_formal=True` | RGB image + motion family + style | official 动态图像套件 | 是 |
- `video_static` | `static/...` videos, `is_formal=True` | RGB video + scene taxonomy + style | official 静态视频套件 | 是 |
- `video_dynamic` | `dynamic/...` videos, `is_formal=True` | RGB video + motion family + style | official 动态视频套件 | 是 |
- `video_nonformal` | `AAA_Games/videos`, `is_formal=False` | RGB video；附加 sidecar 是否齐全待扫描 | auxiliary / experimental | 否 |

在当前数据形态下，**guaranteed supervision** 只有：

1. `category_path`
2. `is_formal`
3. RGB asset 存在
4. static / dynamic
5. photorealistic / stylized
6. indoor / outdoor / 叶子语义类（静态）
7. articulated / deformable / fluid / multi_motion / rigid（动态）

下列监督默认**不保证存在**，必须由 inventory builder 扫描后再决定是否启用：

- segmentation mask
- object track
- camera pose
- instruction text
- event boundary
- physics annotation

### 3.7 基于现有数据的 benchmark 评测范围

这一节给出**官方上线范围**，避免后续 coding agent 按全量 taxonomy 把 benchmark 做得过大。

#### 3.7.1 Official：必须做

official 不再是单一“视频主榜 + 图像辅助榜”，而是四个 formal suite：

| Suite | 数据来源 | guaranteed labels | official 启用能力 |
| --- | --- | --- | --- |
| `WorldArena-Image-Static` | `static/...` images, `is_formal=True` | scene taxonomy + style | scene alignment, style alignment, domain slice |
| `WorldArena-Image-Dynamic` | `dynamic/...` images, `is_formal=True` | motion family + style | dynamic-family alignment, style alignment, optional masked variant |
| `WorldArena-Video-Static` | `static/...` videos, `is_formal=True` | scene taxonomy + style | scene alignment, style alignment, quality, rollout, motion smoothness |
| `WorldArena-Video-Dynamic` | `dynamic/...` videos, `is_formal=True` | motion family + style | dynamic-family alignment, quality, motion magnitude, motion smoothness, rollout |

当前 official 默认只启用**不依赖额外强监督**的指标：

1. `scene_control` / `scene_alignment`
2. `style_consistency`
3. `subjective_quality` / `aesthetic`
4. `motion_magnitude`
5. `motion_smoothness`
6. `rollout_quality_decay`
7. dataset-level distribution metrics，例如 `FVD` / `VBench`-style aggregate（按 suite 单独算）

#### 3.7.2 Conditional official metrics：扫描到 sidecar 才启用

下面这些能力不是默认官方项，只有在 inventory builder 确认相应 sidecar 存在后，才允许升级为 official subset report：

| Metric family | 需要的 sidecar |
| --- | --- |
| `object_control` masked variant | segmentation mask 或 object box |
| `camera_control` proxy | 可稳定恢复的 pose / camera metadata |
| `geom_consistency` proxy | pose / intrinsics / reliable SLAM recovery |
| `long_delayed_effect` | event boundary / instruction / index metadata |
| `basic_physics` | object track 或可稳定追踪的 mask |

这些指标即使启用，也应先作为 **subset report**，不要直接并入全局 `S_worldarena`，直到覆盖率和质量足够稳定。

#### 3.7.3 Auxiliary / experimental：non-formal 数据

`is_formal=False` 的数据默认全部进入 auxiliary / experimental：

- `AAA_Games/videos` 500 条
- image backup folders 共 172 条

它们的用途：

1. metric debug
2. ablation
3. control / geometry / game-world 实验
4. 未来扩展的 prototype set

#### 3.7.4 Future-expansion：明确延期

以下能力在当前数据结构下不应阻塞 benchmark：

1. `long_revisit`
2. 完整 `camera_control` 主榜
3. 完整 `object_control` 主榜
4. 完整 `geom_consistency` 主榜
5. 完整 `interactive`
6. 大规模 `counterfactual / physics` 主榜

### 3.8 现有数据到 benchmark 的构建流程

这一节不讨论“去哪里再找数据”，只讨论**如何把当前数据加工成 benchmark**。

#### 3.8.1 第一步：先跑 inventory，不写死数量

必须先实现一个 manifest builder，例如：

```text
scripts/qa/build_worldarena_inventory.py
```

该脚本的输出应至少包含：

```json
{
  "snapshot_name": "worldarena_freeze_candidate",
  "snapshot_time": "2026-04-21T00:00:00Z",
  "roots": {
    "image_root": "...",
    "video_root": "..."
  },
  "counts": {
    "image_static_formal": {"planned": 0, "scanned": 0, "usable": 0},
    "image_dynamic_formal": {"planned": 0, "scanned": 0, "usable": 0},
    "video_static_formal": {"planned": 0, "scanned": 0, "usable": 0},
    "video_dynamic_formal": {"planned": 0, "scanned": 0, "usable": 0},
    "video_nonformal": {"planned": 0, "scanned": 0, "usable": 0},
    "image_nonformal": {"planned": 0, "scanned": 0, "usable": 0}
  },
  "missing_modalities": {
    "missing_rgb": 0,
    "missing_pose": 0,
    "missing_masks": 0,
    "missing_instruction": 0
  }
}
```

强制要求：

1. 文档中的数量永远只是“规划量级”，不作为代码常量。
2. freeze 时真正使用的数量，以 manifest builder 输出为准。
3. leaderboard manifest 必须记录 snapshot hash 或文件树 hash。
4. manifest builder 还必须校验：`planned_count` 与扫描到的实际样本数是否一致；若不一致，需要输出 `count_mismatch_report.csv`。

#### 3.8.2 Formal data 的标准转换

对所有 `is_formal=True` 的 image / video 资产，统一流程是：

1. 从 `category_path` 恢复：
   - modality: `image` / `video`
   - dynamics: `static` / `dynamic`
   - style: `photorealistic` / `stylized`
   - fine class: 叶子目录名
2. 为每个资产生成标准 `sample.json`
3. 先只填 guaranteed fields：
   - `asset_level`
   - `is_formal`
   - `category_path`
   - `worldscore_tags.dynamic`
   - `worldscore_tags.style`
   - `domain_tags.scene_domain` 或 `physics/motion_family`
4. 对 image 资产生成：
   - `conditioning.png`
5. 对 video 资产生成：
   - `conditioning.png = first frame`
   - `reference_video = full clip`
6. sidecar 若存在，再追加写入：
   - `objects.json`
   - `events.json`
   - `camera_gt.json`
   - `trajectory.json`
   - `extra_sidecars.json`

默认命名：

- `sample_id = {asset_level}_{style}_{fine_class}_{index}`
- `source_name = worldarena_formal_{image|video}`
- `source_group_id = {category_path}::{raw_asset_id}`

#### 3.8.3 Non-formal data 的标准转换

对所有 `is_formal=False` 的资产：

1. 仍然生成标准 `sample.json`
2. 但强制设置：
   - `is_formal = false`
   - `eligible_for_official = false`
   - `suite = auxiliary` 或 `suite = experimental`
3. 如果 sidecar 特别丰富，可以单独挂到 experimental suite，但不得混入 official

#### 3.8.4 Split 与 leaderboard 组织

基于当前资产，benchmark 需要拆成六个报告面板：

1. `WorldArena-Image-Static`
2. `WorldArena-Image-Dynamic`
3. `WorldArena-Video-Static`
4. `WorldArena-Video-Dynamic`

以及一个实验面板：

5. `WorldArena-Auxiliary`
6. `WorldArena-Experimental`

强制规则：

- 只有 `is_formal = true` 的资产进入 official split。
- `video_nonformal` 和 image backup 只进入 auxiliary / experimental。
- `image` 与 `video` 必须分开 split，`static` 与 `dynamic` 也必须分层抽样。
- 不允许把 non-formal 与 formal 样本混成一个无区分总分。

推荐默认总体分数：

```text
S_worldarena =
  0.15 * S_image_static +
  0.15 * S_image_dynamic +
  0.30 * S_video_static +
  0.40 * S_video_dynamic
```

其中：

- `S_video_dynamic` 权重最高，因为它最贴近 world model 的时序建模核心能力
- `S_experimental` 与 `S_auxiliary` 单独报告，不并入 `S_worldarena`

### 3.9 原始数据到 benchmark sample 的标准化加工

本节解决“拿到数据以后，怎样稳定转成 benchmark sample”。

在当前 benchmark 里，**canonical asset 的主来源是四个 formal suite**：

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

`video_nonformal` 只进入 auxiliary / experimental。

#### 3.8.1 统一中间表示

所有来源先转成统一中间表示 `canonical clip`：

```json
{
  "canonical_id": "src_video_dynamic_photorealistic_articulated_0003",
  "source_name": "worldarena_formal_video",
  "source_group_id": "dynamic/photorealistic/videos/articulated::0003",
  "start_time_sec": 8.0,
  "end_time_sec": 18.0,
  "fps_native": 30.0,
  "resolution_native": [1280, 720],
  "path_video": "raw_clip.mp4",
  "path_frames": "frames/",
  "path_audio": null,
  "path_pose": null,
  "path_actions": null,
  "path_calib": null
}
```

只有 canonical clip 才允许进入后续 mining。

#### 3.8.2 自动挖掘前必须先计算的共用缓存

每个 canonical clip 都先算：

1. `CLIP` anchor embeddings
2. `DINOv2` anchor embeddings
3. `TransNetV2` cut points
4. `DROID-SLAM` pose（如果可解）
5. `GroundingDINO + SAM2` 候选物体轨迹
6. `Depth Anything V2` depth（无 GT depth 时）
7. `RAFT / GMFlow` 光流统计

缓存路径建议：

```text
artifacts/{canonical_id}/
  clip_meta.json
  clip_clip_embeddings.pt
  clip_dino_embeddings.pt
  cut_points.json
  slam_pose.json
  object_tracks.json
  depth/
  flow/
```

#### 3.8.3 Track 候选挖掘规则

##### A. `camera_control` 候选

这一类候选**默认不是 official 主项**。只有在 inventory builder 扫描确认某个 subset 具备稳定的 pose / camera metadata，或自动 SLAM 恢复成功率足够高时，才启用。

先基于 pose 计算每帧相对位姿：

```math
T'_t = T_0^{-1} T_t
```

设平移分量为 `p_t = (x_t, y_t, z_t)`，旋转分量转为 `yaw_t, pitch_t, roll_t`。

定义每轴运动能量：

```math
m_x = \text{median}_t |x_t - x_{t-1}|,\;
m_y = \text{median}_t |y_t - y_{t-1}|,\;
m_z = \text{median}_t |z_t - z_{t-1}|
```

```math
m_{yaw} = \text{median}_t |\Delta yaw_t|,\;
m_{pitch} = \text{median}_t |\Delta pitch_t|,\;
m_{roll} = \text{median}_t |\Delta roll_t|
```

定义 dominant ratio：

```math
r_{dom} = \frac{\max(m_x,m_y,m_z,m_{yaw},m_{pitch},m_{roll})}{m_x+m_y+m_z+m_{yaw}+m_{pitch}+m_{roll}+\epsilon}
```

保留规则：

- `r_slam = N_valid_pose / F >= 0.8`
- `r_dom >= 0.45`
- `r_cut = 0`

然后根据 dominant axis 自动生成 `camera_instruction_text` 初稿：

- `+x` / `-x`：left / right truck
- `+z` / `-z`：push-in / pull-out
- `yaw`：pan left / pan right
- `pitch`：tilt up / tilt down

最后人工只做文本修正，不重新从零写。

##### B. `long_revisit` 候选

这类候选在当前数据结构下**不属于 official**。只有当后续确实存在长时 loop-closure 或同一 scene 回访序列时，才启用这一条。

从一个较长 clip 中搜索 `i < j`，要求：

- `j - i >= max(24, 0.6F)`
- `cos(dino_i, dino_j)` 高
- 如果有 pose，则回环位置接近

定义候选分数：

```math
S_{revisit}(i,j)=0.45\cos(d_i,d_j)+0.35\cos(c_i,c_j)+0.20\exp\left(-\frac{\|p_i-p_j\|_2^2}{\sigma_p^2}\right)
```

其中：

- `d_i, d_j`：DINOv2 embedding
- `c_i, c_j`：CLIP embedding
- `p_i, p_j`：pose 平移

若没有 pose，则去掉最后一项并上人工复核。

保留规则：

- `max_{i,j} S_revisit(i,j) >= 0.82`
- 起点和回访点都不是 cut 附近帧
- 中间至少经历一个明显的非重复区段

##### C. `long_delayed_effect` 候选

在当前 benchmark 中，这一条默认归入 `experimental`，只在某个 subset 具备可验证的 event boundary / staged progression 信息时启用。

定义两个时间点：

- `t_event`：早期事件发生时间
- `t_effect`：晚期可观察结果首次稳定出现时间

定义 delay：

```math
\Delta_{delay} = t_{effect} - t_{event}
```

保留规则：

- `t_event <= 0.5F`
- `\Delta_delay >= max(12, 0.15F)`
- effect 在后续至少持续 `0.2F` 帧

优先顺序：

1. 如果 sidecar metadata 能直接给出阶段边界，就优先用它。
2. 如果没有，就用 `GroundingDINO + SAM2` 跟踪关键对象，再由人工确认 `events.json`。

##### D. `object_control` 候选

对目标 object 运行：

1. `GroundingDINO` 文本检测
2. `SAM2` 初始化 mask
3. `SAM2` 视频跟踪

定义：

```math
r_{obj} = \frac{1}{F}\sum_t \mathbf{1}[mask_t\;\text{valid}]
```

```math
a_{obj} = \text{median}_t \frac{\text{area}(mask_t)}{W \cdot H}
```

保留规则：

- `r_obj >= 0.6`
- `0.01 <= a_obj <= 0.5`
- 目标物体类别明确且跨帧 identity 稳定

##### E. `physics` 候选

对 primary object 中心轨迹 `x_t` 做模型拟合。以抛体为例：

```math
x(t) = v_x t + x_0
```

```math
y(t) = \frac{1}{2} g t^2 + v_y t + y_0
```

定义归一化拟合误差：

```math
e_{fit} = \frac{\sqrt{\frac{1}{F}\sum_t \| \hat{p}_t - p_t \|_2^2}}{\text{diag}(bbox_{median})}
```

定义拟合分数：

```math
S_{fit} = \max(0, 1 - e_{fit})
```

保留规则：

- `r_obj >= 0.7`
- `S_fit >= 0.75`
- 运动家族可以归入 `projectile / slide / pendulum / collision / fluid-like`

##### F. `geom_consistency` 候选

保留规则：

- 多视角同步，或
- `r_slam >= 0.7` 且 `N_keyframes >= 12`

优先来源：

- 已确认 pose / intrinsics 存在的 formal video subset
- 自动 SLAM 恢复稳定的 video subset

#### 3.8.4 从候选 clip 生成最终 sample

对每条通过筛选的 canonical clip，生成 benchmark sample：

##### 协议 A. `i2v`

1. `conditioning.png = frame_0`
2. `reference_video = full clip`
3. `text_prompt_current`：描述 conditioning frame
4. `text_prompt_target`：描述参考视频的目标世界演化

##### 协议 B. `video continuation`

1. `conditioning.mp4 = 前 K 帧`
2. `reference_video = 后续帧`
3. `num_frames` 只统计待生成部分

##### 协议 C. `action-conditioned`

1. `conditioning.mp4` 或 `conditioning.png`
2. `action.json`
3. `reference_video`
4. 如果有 coarse / fine action，都要保留

### 3.10 标注升级策略

核心原则：**能自动预填的全部自动预填，人工只做验证和纠错。**

#### 3.9.1 自动预填内容

| 字段 | 自动来源 |
| --- | --- |
| `scene_domain` / `style_domain` / `pov` | 来源 metadata + CLIP tagger |
| `text_prompt_current` | 第一帧 captioner / VLM 草稿 |
| `text_prompt_target` | clip-level caption + track 模板 |
| `camera_instruction_text` | pose dominant axis 模板 |
| `objects.json` | GroundingDINO + SAM2 + noun extraction |
| `events.json` | action log 或 candidate mining |
| `physics.json.motion_family` | trajectory fit family |

#### 3.9.2 人工必须确认的内容

1. track 分配是否正确
2. prompt 是否准确、是否泄露 reference 细节
3. target object 名称是否统一
4. delayed effect 的 `t_event / t_effect`
5. physics family 与主对象是否正确
6. 是否存在内容风险、版权风险或明显错误标注

#### 3.9.3 Gold / Silver 样本分级

- `gold`：人工逐条复核，允许进入 hidden test
- `silver`：自动预填 + spot check，可进入 dev

强制规则：

- hidden leaderboard set 只能由 `gold` 样本组成
- `physics`、`counterfactual`、`long_delayed_effect` 默认必须 `gold`

### 3.11 数据切分、去重与泄漏控制

benchmark 最容易出问题的不是 metric，而是数据泄漏。

#### 3.10.1 split 的基本原则

1. 同一个 `source_group_id` 绝不跨 split。
2. 同一 raw asset、同一语义叶子目录簇、同一视频切片 parent、同一 non-formal group 绝不跨 split。
3. near-duplicate 也不能跨 split。

#### 3.10.2 embedding 泄漏检查

定义 cross-split leakage：

```math
\text{leak}(i,j) = \mathbf{1}\left[\cos(e^{clip}_i,e^{clip}_j)>0.98 \lor \cos(e^{dino}_i,e^{dino}_j)>0.97\right]
```

所有 `leak(i,j)=1` 的跨 split pair 都要进入人工复核队列。

#### 3.10.3 推荐 split 结构

如果 benchmark 主要用于 evaluation，而不是给训练集，建议用：

- `dev_public`: 20%
- `dev_private`: 10%
- `test_hidden`: 70%

原因：

- benchmark 的主要价值在 hidden test，而不是公开训练数据。
- `dev_public` 只需要足够做 sanity check 和本地 smoke test。

### 3.12 冷启动建设路线

这里的冷启动不再指“去外面收更多数据”，而是指：**如何用当前 WorldArena 数据先冻结一个能工作的 benchmark。**

#### Phase 0: 先冻结 inventory 与可用性

目标：

1. 跑出 `worldarena_inventory.json`
2. 明确五类资产：`image_static`、`image_dynamic`、`video_static`、`video_dynamic`、`video_nonformal`
3. 明确每类资产里真正可用的样本数
4. 明确缺模态统计：缺 RGB、缺 pose、缺 mask、缺 instruction

交付物：

- `inventory/worldarena_inventory.json`
- `inventory/worldarena_inventory.csv`
- `inventory/missing_modalities_report.json`

#### Phase 1: 做出 official

目标：只基于当前可完整闭环的数据做主榜。

包含：

1. `WorldArena-Image-Static`
2. `WorldArena-Image-Dynamic`
3. `WorldArena-Video-Static`
4. `WorldArena-Video-Dynamic`

此阶段的关键不是样本绝对数量，而是：

- sample schema 跑通
- evaluator 跑通
- per-suite 报表跑通
- split / 去重 / hidden test 机制跑通

#### Phase 2: 打开 experimental

包含：

1. subset-level `camera_control`
2. subset-level `object_control`
3. `long_delayed_effect`
4. `basic physics`
5. `video_nonformal` auxiliary

这一步依赖的是**当前数据内部的进一步整理**，而不是新增外部来源。

#### Phase 3: 未来扩展

只有在 official 已经稳定以后，才考虑新增外部来源或新赛道，例如：

1. `long_revisit`
2. 完整 `interactive`
3. 更高质量的 geometry/control 子榜
4. 多来源混合 leaderboard

### 3.13 当前 benchmark 明确不建议做的事

1. **不要为了当前 benchmark 去重新设计一个依赖外部数据的大而全 benchmark。**
2. **不要把 `videos/dynamic` 默认当成“完整视频集”使用，除非 RGB / annotation 已验证对齐。**
3. **不要把 image-only 数据混进 video 主榜总分。**
4. **不要把 experimental 轨道并入 official 总分。**
5. **不要把当前探索阶段看到的数量写死到代码和文档。**
6. **不要只做自动清洗、不做人审就冻结 hidden test。**

## 4. 统一预处理与缓存

所有 metric 共享同一套预处理，避免重复算：

### 4.1 Frame decode

1. 用 `ffmpeg` 或 `decord` 把视频解成 frame。
2. 所有 sample 统一到 benchmark target FPS，例如 `24 FPS`。
3. 只在需要时 resize；默认保留原分辨率，metric 内部各自做 resize。

输出：

```text
artifacts/{sample_id}/frames/000000.png
artifacts/{sample_id}/frames/000001.png
...
```

### 4.2 Anchor frame 采样

定义 anchor frame 集合：

- `A0 = 0`
- `A1 = floor((F - 1) / 3)`
- `A2 = floor(2 * (F - 1) / 3)`
- `A3 = F - 1`

对于 revisit 类任务，额外提供 `revisit_anchor_idx`，通常是回到起点时的 frame。

对于长视频，可增加：

- `stride_anchor = max(8, floor(F / 12))`
- `dense_anchor = [0, stride_anchor, 2*stride_anchor, ...]`

### 4.3 共用缓存

每个 sample 至少缓存：

- `camera_pose_est.json`
- `optical_flow/`
- `groundingdino_detections.json`
- `sam2_tracks.json`
- `depth_anything_depth/`
- `met3r_pairs.json`

### 4.4 推荐底层模型

统一推荐下面这些实现，优先选择已经有公开代码、复现成本低、和 WorldScore / MEt3R / Morpheus 一致的组合。

| 用途 | 模型 / 算法 | 推荐实现 |
| --- | --- | --- |
| Camera pose estimation | DROID-SLAM | 与 WorldScore 一致 |
| Open-vocabulary object detect | GroundingDINO SwinT OGC | 与 WorldScore 一致 |
| Zero-shot segmentation / tracking | SAM2.1 Hiera B+ | 与 WorldScore / Morpheus 一致 |
| Optical flow | SEA-RAFT | 与 WorldScore 一致 |
| 3D pair consistency | MEt3R + DUSt3R / MASt3R + FeatUp DINO | MEt3R 官方实现 |
| Depth sanity check | Depth Anything V2 | 与 Morpheus 一致 |
| Text-image alignment | CLIPScore, `openai/clip-vit-base-patch16` | WorldScore |
| Subjective quality | CLIP-IQA+ | WorldScore / DrivingGen |
| Aesthetic quality | CLIP + MLP aesthetic predictor | WorldScore |
| Motion smoothness interpolation | VFIMamba | WorldScore |
| Style consistency | VGG19 Gram Matrix | WorldScore |

## 5. Score Normalization 统一规则

原始 metric 的量纲不同，必须先归一化再聚合。

### 5.1 对 lower-is-better error metric

如果 metric 原始值为 `x`，经验上下界为 `[l, u]`，且越小越好：

```text
x_clip = min(max(x, l), u)
s_norm = 1 - (x_clip - l) / (u - l)
```

适用：

- reprojection error
- optical-flow AEPE
- MEt3R dissimilarity
- Gram matrix style loss
- ADE / DTW
- trajectory NMSE

### 5.2 对 higher-is-better bounded metric

```text
x_clip = min(max(x, l), u)
s_norm = (x_clip - l) / (u - l)
```

适用：

- object detection rate
- DINO cosine similarity
- SSIM
- IoU
- CLIP-IQA+（如果不做 z-score）

### 5.3 对开放区间 metric 使用 z-score mapping

对 WorldScore 风格的开放值 metric，使用：

```text
z = (x - mean_ref) / std_ref
z_clip = min(max(z, z_min), z_max)
s_norm = r0 + (r1 - r0) * (z_clip - z_min) / (z_max - z_min)
```

默认：

- `r0 = 0.25`
- `r1 = 0.75`

这样做的原因：

- 避免开放值 metric 一票否决
- 保持与 WorldScore 相近的分布压缩方式

适用：

- CLIPScore
- CLIP-IQA+
- Aesthetic score
- motion magnitude

### 5.4 经验上下界怎么定

不要手工拍脑袋，统一按 benchmark dev split 自动估计：

1. `empirical_best`
   - real reference vs real reference 同类配对
   - 或者真实视频自身的 metric
2. `empirical_worst`
   - 不同样本随机配对
   - 打乱 frame 顺序
   - 打乱 style / object prompt
   - 静止摄像机、空动作、插值视频等坏 baseline

建议把 bounds 固定在 `dev split` 上，写入：

```text
configs/worldmodel_metric_bounds.json
```

## 6. Metric Catalog

---

## 6.1 Long-range Static Revisit Consistency

### 6.1.1 适用场景

适用于：

- 房间 / 街区 / 室外场景绕行
- 相机离开起点后长时间运动，再回到早先位置

### 6.1.2 输入

- 生成视频 `V_gen`
- 参考视频 `V_ref`
- `camera_gt.json`（推荐）
- `revisit_anchor_idx`
- `objects.json`

### 6.1.3 原始 metric

对该 track，不建议只用单一 metric。建议同时算 4 个子项：

1. `Revisit-Pose Error`
2. `Revisit-MEt3R`
3. `Revisit-DINO Similarity`
4. `Revisit-Object F1`

### 6.1.4 Revisit-Pose Error

如果有 GT camera，使用 DROID-SLAM 估计 pose，再与 GT 对齐，公式与 WorldScore 相同。

设：

- `R_pred_t, t_pred_t` 为估计相机旋转和平移
- `R_gt_t, t_gt_t` 为 GT 相机旋转和平移
- `s` 为通过最小化平移误差求出的尺度因子

旋转误差：

```text
R_err_t = arccos((trace(R_pred_t R_gt_t^T) - 1) / 2) * 180 / pi
```

平移误差：

```text
T_err_t = || s * t_pred_t - t_gt_t ||_2
```

对 revisit 任务只统计：

- 起点 anchor
- revisit anchor
- 以及两者中间的关键帧

最终：

```text
R_err = mean_t R_err_t
T_err = mean_t T_err_t
S_pose = geometric_mean(norm(R_err), norm(T_err))
```

其中 `norm()` 按第 5 节 lower-is-better 规则归一化。

### 6.1.5 Revisit-MEt3R

MEt3R 用于测量“回到相同视角后，几何是否一致”。

使用：

- backbone: `dust3r` 或 `mast3r`
- feature backbone: `dino16`
- upsampler: `featup`
- distance: `cosine`

对于同一对图像 `I_a`（起点 anchor）和 `I_r`（revisit anchor）：

1. 用 DUSt3R / MASt3R 估计两视图 dense 3D point map
2. 用 FeatUp-DINO 取高分辨率 feature
3. 把 feature unproject 到 3D point cloud
4. 用 PyTorch3D rasterize 到同一 camera frame
5. 计算 overlap mask
6. 在 overlap 区域上计算 feature dissimilarity

MEt3R 的核心像素级 dissimilarity：

```text
d(p) = 1 - <f1(p), f2(p)> / (||f1(p)|| * ||f2(p)|| + eps)
```

全图分数：

```text
MEt3R(I_a, I_r) = sum_p d(p) * m(p) / sum_p m(p)
```

其中 `m(p)` 是 overlap mask。

对于 revisit 任务：

```text
S_met3r = 1 - norm(MEt3R(I_a, I_r))
```

`norm()` 用 dev split 的 `real pair` 和 `random pair` 估计上下界。

### 6.1.6 Revisit-DINO Similarity

用 DINOv2 embedding 测 anchor 与 revisit 图像的高层语义一致性。

公式：

```text
f1 = DINOv2(I_a)
f2 = DINOv2(I_r)
S_dino = cosine(f1, f2)
```

注意：

- 这是 appearance / semantic consistency，不是 geometry metric
- 它不能替代 MEt3R，只能作为补充

### 6.1.7 Revisit-Object F1

步骤：

1. 从 `objects.json` 读取 expected object list
2. 用 GroundingDINO 在 `I_a` 和 `I_r` 上检测对象
3. 对每个 expected object 做 presence matching
4. 计算 precision / recall / F1

定义：

```text
Precision = TP / (TP + FP + eps)
Recall = TP / (TP + FN + eps)
F1 = 2PR / (P + R + eps)
```

matching 规则：

- 先 exact string match
- 再 noun-level partial match
- 与 WorldScore object metric 的 matching 策略一致

### 6.1.8 长时静态一致性总分

```text
S_long_revisit =
  0.35 * S_pose +
  0.35 * S_met3r +
  0.20 * S_dino +
  0.10 * S_object_f1
```

如果没有 GT camera：

```text
S_long_revisit_no_gt =
  0.50 * S_met3r +
  0.30 * S_dino +
  0.20 * S_object_f1
```

---

## 6.2 Long-range Delayed-effect Prediction

### 6.2.1 适用场景

适用于：

- 早期动作在晚期才显现
- 遮挡后恢复
- 事件驱动变化（碰撞后反弹、开门后场景变化、拉杆后后续运动）

### 6.2.2 输入

- `events.json`
- 生成视频 `V_gen`
- 参考视频 `V_ref`
- 可选对象 mask / ROI

### 6.2.3 子 metric

1. `Event Time Error`
2. `Trajectory NMSE`
3. `Post-event State Accuracy`

### 6.2.4 Event Time Error

设参考视频中事件发生在 `t_ref`，生成视频估计事件发生在 `t_gen`。

```text
ETE = |t_gen - t_ref|
S_event = 1 - norm(ETE)
```

事件时间检测规则：

- 对 collision：目标 object 加速度或速度符号改变时刻
- 对 occlusion exit：mask area 从 0 恢复为稳定正值时刻
- 对 switch / light change：ROI 平均亮度过阈值时刻

### 6.2.5 Trajectory NMSE

如果 reference 中有可比轨迹，先抽取主对象轨迹。

设真实轨迹为 `y_t`，生成轨迹为 `y_hat_t`：

```text
MSE = (1 / T) * sum_t ||y_hat_t - y_t||_2^2
Var = (1 / T) * sum_t ||y_t - mean(y)||_2^2
NMSE = MSE / (Var + eps)
S_traj = max(1 - NMSE, 0)
```

### 6.2.6 Post-event State Accuracy

只看事件发生后的稳定区间 `W_post`。

例子：

- 碰撞后向左 / 向右
- 开门后门是开 / 关
- 物体是否还存在

如果状态是离散标签：

```text
S_post = accuracy(y_post_pred, y_post_gt)
```

如果状态是连续量：

```text
S_post = 1 - norm(mean_{t in W_post} ||s_hat_t - s_t||_2)
```

### 6.2.7 长时 delayed-effect 总分

```text
S_long_delayed =
  0.30 * S_event +
  0.40 * S_traj +
  0.30 * S_post
```

---

## 6.3 Camera Controllability

### 6.3.1 三种输入模式

必须分开实现，不能混在一起：

1. **trajectory-supervised**
   - 输入完整相机轨迹 / camera matrix
2. **text-supervised**
   - 输入自然语言镜头指令
3. **action-supervised**
   - 输入键盘 / 鼠标 / joystick low-level action

### 6.3.2 Trajectory-supervised：Pose Error

直接复用第 6.1.4 的 `R_err` / `T_err`：

```text
S_cam_traj = geometric_mean(norm(R_err), norm(T_err))
```

### 6.3.3 Text-supervised：Primitive Accuracy

先把文本指令 parse 成标准 primitive vector。

统一 primitive 空间：

```text
[yaw_left_right, pitch_up_down, roll_ccw_cw, tx_left_right, ty_up_down, tz_forward_backward]
```

例如：

- `pan left` -> `[+1, 0, 0, 0, 0, 0]`
- `move left` -> `[0, 0, 0, +1, 0, 0]`
- `push in` -> `[0, 0, 0, 0, 0, +1]`
- `pull out` -> `[0, 0, 0, 0, 0, -1]`
- `orbit left` -> `[+1, 0, 0, +1, 0, 0]`（绕主体左转并伴随横向位移）

然后：

1. 用 DROID-SLAM 估计相机 pose 序列
2. 转成每段 `delta pose`
3. 把 `delta pose` 映射为 primitive vector `u_k`

其中：

```text
u_k = [sign(Delta yaw), sign(Delta pitch), sign(Delta roll),
       sign(Delta x), sign(Delta y), sign(Delta z)]
```

对每个要求非零的维度：

```text
A_k = mean_i 1[sign(u_k[i]) == c_k[i]]
```

对幅度：

```text
M_k = mean_i min(|delta_i| / tau_i, 1.0)
```

默认阈值：

- `tau_yaw = 5 deg`
- `tau_pitch = 5 deg`
- `tau_roll = 5 deg`
- `tau_translation = 0.05 * scene_scale`

最终：

```text
S_cam_text = 0.7 * mean_k A_k + 0.3 * mean_k M_k
```

### 6.3.4 Action-supervised：Low-level Action Alignment

动作序列统一定义为：

```json
[
  {"t0": 0, "t1": 12, "action": "key_w"},
  {"t0": 13, "t1": 20, "action": "mouse_dx_pos"},
  {"t0": 21, "t1": 28, "action": "mouse_dx_neg"}
]
```

映射：

- `key_w` -> `tz_forward`
- `key_s` -> `tz_backward`
- `key_a` -> `tx_left`
- `key_d` -> `tx_right`
- `mouse_dx_pos` -> `yaw_right`
- `mouse_dx_neg` -> `yaw_left`
- `mouse_dy_pos` -> `pitch_down`
- `mouse_dy_neg` -> `pitch_up`

对每个 action segment 统计：

```text
S_seg = 0.5 * direction_accuracy + 0.5 * magnitude_satisfaction
```

整个视频：

```text
S_cam_action = mean_seg S_seg
```

### 6.3.5 Camera score 聚合

按输入模式三选一：

- trajectory -> `S_cam_traj`
- text -> `S_cam_text`
- action -> `S_cam_action`

---

## 6.4 Object Controllability

### 6.4.1 输入

- 目标对象列表
- 可选：目标对象初始 mask
- 可选：目标对象目标区域 / 目标事件

### 6.4.2 Object Presence Score

与 WorldScore 一致，基础版用 GroundingDINO detection rate。

对于每个 expected object：

1. 在 anchor frame 上用 GroundingDINO 检测
2. 如果检测到与目标 noun 匹配的短语，计为成功

单帧：

```text
presence_rate = matched_objects / expected_objects
```

视频级：

```text
S_obj_presence = mean_anchor presence_rate
```

### 6.4.3 Object Persistence Score

对需要“保持身份不漂移”的对象，使用 SAM2 track。

设对象初始 mask 为 `M_0`，第 `t` 帧追踪 mask 为 `M_t`。

定义：

- `area_ratio_t = area(M_t) / area(M_0)`
- `center_disp_t = ||center(M_t) - expected_center_t||`

如果任务只要求“对象持续存在”，则：

```text
S_persist = (# frames with area_ratio_t > tau_area) / T
```

默认 `tau_area = 0.2`

### 6.4.4 Object Action Response Score

用于 object keyboard control 或 interaction task。

先抽取对象轨迹 `p_t`，再根据动作类型定义 observable。

例子：

- `push_left` -> 要求 `x_end - x_start < 0`
- `lift_up` -> 要求 `y_end - y_start < 0`
- `spawn_object` -> presence 由 0 变 1

定义：

```text
direction_ok = 1[sign(obs_gen) == sign(obs_expected)]
magnitude_ok = min(|obs_gen| / (|obs_ref| + eps), 1)
S_obj_action = 0.5 * direction_ok + 0.5 * magnitude_ok
```

### 6.4.5 Object control 聚合

```text
S_object_control =
  0.40 * S_obj_presence +
  0.30 * S_persist +
  0.30 * S_obj_action
```

如果某个任务没有 action，则把 `S_obj_action` 权重均分给其他项。

---

## 6.5 Scene / Content Alignment

### 6.5.1 主 metric：CLIPScore

与 WorldScore 一致，使用：

- model: `openai/clip-vit-base-patch16`

公式：

```text
CLIPScore(I, p) = max(100 * cosine(E_image(I), E_text(p)), 0)
```

视频级：

```text
S_clip = mean_anchor CLIPScore(I_anchor, target_prompt)
```

### 6.5.2 辅助 metric：Object Set Recall

如果 target scene prompt 中可抽出显式 noun 列表，则再算 object set recall：

```text
Recall_obj = matched_target_nouns / target_nouns
```

### 6.5.3 Scene control 聚合

```text
S_scene_control =
  0.80 * norm(S_clip) +
  0.20 * Recall_obj
```

如果没有 object list，则只报告 `norm(S_clip)`。

---

## 6.6 3D / 4D Consistency

### 6.6.1 3D consistency：MEt3R

对任意 frame pair `(I_i, I_j)`：

```text
S_3d_pair = 1 - norm(MEt3R(I_i, I_j))
```

pair 选择策略：

1. anchor pair：`(A0, A1), (A1, A2), (A2, A3)`
2. revisit pair：`(A0, A_revisit)`（若存在）
3. dense pair：stride 为 8 或 12 的相邻 pair

最终：

```text
S_3d_consistency = mean_pairs S_3d_pair
```

### 6.6.2 4D temporal geometry：Reprojection Error

使用 DROID-SLAM 在整段视频上计算 valid reprojection error：

```text
E_reproj = mean(valid_reprojection_errors)
S_4d_reproj = 1 - norm(E_reproj)
```

### 6.6.3 4D score 聚合

```text
S_geom_consistency =
  0.65 * S_3d_consistency +
  0.35 * S_4d_reproj
```

---

## 6.7 Photometric Consistency

### 6.7.1 原理

WorldScore 的 photometric consistency 实现本质是**forward-backward optical flow cycle consistency**。

### 6.7.2 模型

- optical flow: `SEA-RAFT`

### 6.7.3 公式

给定相邻帧 `I_t, I_{t+1}`：

- `F12`：从 `I_t -> I_{t+1}` 的光流
- `F21`：从 `I_{t+1} -> I_t` 的光流

对像素 `p = (x, y)`：

```text
p'  = p + F12(p)
p'' = p' + F21(round(p'))
EPE(p) = ||p'' - p||_2
```

单对帧：

```text
AEPE_t = mean_p EPE(p)
```

整段视频：

```text
AEPE = mean_t AEPE_t
S_photo = 1 - norm(AEPE)
```

### 6.7.4 实现步骤

1. 解出全视频 frames
2. 对每一对相邻帧跑 SEA-RAFT forward 和 backward flow
3. 只在中心 crop 上算 AEPE，避免边界噪声
4. 对所有相邻帧取平均

### 6.7.5 注意事项

- 这个 metric 衡量的是相邻帧光度/运动一致性，不衡量全局几何
- 所以它必须和 MEt3R / reprojection error 分开报告

---

## 6.8 Style Consistency

### 6.8.1 原理

用 VGG19 feature 的 Gram matrix 测量 style drift。

### 6.8.2 模型

- `torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)`

### 6.8.3 公式

对某层 feature map `F_l(x) in R^{C_l x H_l x W_l}`：

```text
G_l(x) = F_l(x) F_l(x)^T / (C_l H_l W_l)
```

style loss：

```text
L_style(x_ref, x_pred) = sum_{l in L} ||G_l(x_ref) - G_l(x_pred)||_F^2
```

其中 `L = {conv1, conv2, conv3, conv4, conv5}`。

视频级：

```text
S_style = 1 - norm(mean_anchor L_style)
```

### 6.8.4 参考图像

reference image 一律使用：

- conditioning image
- 或 reference video 第一帧

---

## 6.9 Subjective Quality

### 6.9.1 CLIP-IQA+

使用 `iqa-pytorch` 的 `clipiqa+`。

它本质上把图像与若干正负 prompt 做比较，如：

- `Good image / bad image`
- `Sharp image / blurry image`
- `High resolution image / low resolution image`
- `Noise-free image / noisy image`

对视频：

```text
S_iqa = mean_anchor CLIP-IQA+(I_anchor)
```

### 6.9.2 CLIP Aesthetic

使用 `CLIP ViT-L/14` image embedding + MLP aesthetic predictor。

视频：

```text
S_aesthetic = mean_anchor Aesthetic(I_anchor)
```

### 6.9.3 Subjective Quality 聚合

```text
S_subjective =
  0.60 * norm(S_iqa) +
  0.40 * norm(S_aesthetic)
```

---

## 6.10 Motion Magnitude

### 6.10.1 原理

与 WorldScore 一致，用 optical flow magnitude 的中位数衡量“到底有没有在动”。

### 6.10.2 公式

对相邻帧 flow `F_t(p) = (u_t(p), v_t(p))`：

```text
m_t(p) = sqrt(u_t(p)^2 + v_t(p)^2)
M_t = median_p m_t(p)
```

整段视频：

```text
M = mean_t M_t
S_motion_mag = norm(M)
```

这个分数不是越大越好到无穷，而是先按 z-score 压到 `[0.25, 0.75]`，避免模型只靠乱动刷分。

---

## 6.11 Motion Smoothness

### 6.11.1 原理

与 WorldScore 一致，采用“丢掉奇数帧，再插值恢复”的方式衡量运动是否自然。

### 6.11.2 模型

- frame interpolation: `VFIMamba`

### 6.11.3 步骤

1. 原视频帧序列 `I_0, I_1, ..., I_{T-1}`
2. 保留偶数帧：`I_0, I_2, I_4, ...`
3. 对每对偶数帧 `(I_2k, I_2k+2)` 插值得到 `I_hat_2k+1`
4. 与真实奇数帧 `I_2k+1` 比较

### 6.11.4 公式

```text
MSE_k   = mean((I_hat_2k+1 - I_2k+1)^2)
SSIM_k  = SSIM(I_hat_2k+1, I_2k+1)
LPIPS_k = LPIPS(I_hat_2k+1, I_2k+1)
```

整段：

```text
MSE   = mean_k MSE_k
SSIM  = mean_k SSIM_k
LPIPS = mean_k LPIPS_k
```

归一化后：

```text
S_smooth = (norm_inv(MSE) + norm(SSIM) + norm_inv(LPIPS)) / 3
```

其中 `norm_inv()` 表示 lower-is-better 归一化。

---

## 6.12 Basic Physics Consistency

### 6.12.1 适用对象

优先做下面这些 closed-form motion family：

- uniform motion
- constant acceleration
- deceleration
- projectile
- slope sliding
- circular motion
- rigid rotation
- damped oscillation
- pendulum

### 6.12.2 共用预处理：轨迹抽取

对于所有 physics sample，统一流程：

1. 从 `physics.json` 读取 primary object
2. 如果有初始 mask / point prompt，直接喂给 SAM2
3. 如果没有，先用 GroundingDINO 找 object，再用 SAM2 初始化
4. 每帧得到 object mask `M_t`
5. 中心点：

```text
c_t = (1 / |M_t|) * sum_{p in M_t} p
```

6. 若需要深度 sanity check，则对每帧跑 Depth Anything V2，取 mask 内平均深度 `d_t`
7. 速度用 central difference + regression + Savitzky-Golay 平滑

### 6.12.3 速度与加速度估计

设中心轨迹为 `p_t`，帧间隔 `Delta t = 1 / fps`。

central difference 速度：

```text
v_t = (p_{t+1} - p_{t-1}) / (2 Delta t),   1 <= t <= T-2
```

端点：

```text
v_0     = (p_1 - p_0) / Delta t
v_{T-1} = (p_{T-1} - p_{T-2}) / Delta t
```

central difference 加速度：

```text
a_t = (p_{t+1} - 2 p_t + p_{t-1}) / Delta t^2
```

为了抑制 tracking noise，建议再做：

1. 滑动窗口线性回归估计局部斜率
2. central difference 与 regression 结果做加权平均
3. 对速度 / 加速度施加 Savitzky-Golay filter

默认参数：

- regression window `w = 7`
- Savitzky-Golay order `= 3`
- regression 权重 `alpha = 0.7`

### 6.12.4 Discard Rule

必须先过滤根本无法谈物理的视频。建议保留 Morpheus 的三个 discard 原因，并把阈值写死到 config：

#### disappear

若满足任一条件：

- `mask area < 0.1 * area_0` 持续超过 `max(3, 0.1T)` 帧
- tracker confidence 低于阈值的帧占比 > `0.2`

则记为 `discard_disappear = 1`

#### duplicate

对每帧用 GroundingDINO + tracking 估计目标对象数量 `n_t`。

若：

```text
mean_t 1[n_t != expected_count] > 0.2
```

则记为 `discard_duplicate = 1`

#### still

若：

```text
median_t ||v_t||_2 < 2 px/frame
and
||p_{T-1} - p_0||_2 < 0.05 * image_diagonal
```

则记为 `discard_still = 1`

### 6.12.5 Discard Rate

对一个 model 在一个 physics subset 上：

```text
DiscardRate = (N_disappear + N_duplicate + N_still) / N_total
S_valid = 1 - DiscardRate
```

### 6.12.6 Dynamical Score

默认实现不要直接上 PINN，先做**解析模型拟合**，更稳定，也更适合 benchmark 工程化。

#### A. Constant velocity

```text
p_hat(t) = p0 + v t
```

#### B. Constant acceleration / projectile

```text
x_hat(t) = x0 + vx t
y_hat(t) = y0 + vy t + 0.5 a t^2
```

#### C. Circular motion

```text
x_hat(t) = cx + r cos(omega t + phi)
y_hat(t) = cy + r sin(omega t + phi)
```

#### D. Damped oscillation / pendulum small-angle

```text
theta_hat(t) = A exp(-beta t) cos(omega t + phi) + c
```

参数通过最小二乘或 `scipy.optimize.least_squares` 拟合。

拟合后：

```text
MSE  = (1 / T) * sum_t ||p_hat(t) - p_t||_2^2
Var  = (1 / T) * sum_t ||p_t - mean(p)||_2^2
NMSE = MSE / (Var + eps)
S_dyn = max(1 - NMSE, 0)
```

### 6.12.7 PINN Advanced Mode

对于双摆或没有稳定 closed-form 的系统，再启用 Morpheus 风格 PINN：

```text
L_total = L_data + lambda_phy * L_physics
```

默认网络：

- 2 hidden layers
- 20 neurons each
- activation: `tanh`
- optimizer: Adam
- `lr = 1e-3`
- `steps = 2e5`

但 benchmark 第一版不建议把 PINN 作为默认路径，只把它做成 `--physics-fit-mode pinn` 可选项。

---

## 6.13 Physical Invariance Score

### 6.13.1 原理

Morpheus 的核心思想是：不要求生成轨迹和某个 reference 一模一样，而是要求**守恒量在时间上稳定**。

### 6.13.2 各 motion family 对应 invariant

#### Falling ball / projectile / bouncing ball

- total energy
- vertical acceleration
- horizontal momentum

定义：

```text
E_t = 0.5 * (v_x(t)^2 + v_y(t)^2) + g * y_t
a_y(t) = d^2 y / dt^2
p_x(t) = v_x(t)
```

如果没有绝对尺度，则先做 scale normalization：

- 用 reference real subset 估计同类运动的 `g_pixel`
- 或把 `g` 看成拟合参数，只看 invariant 的相对稳定性

#### Holonomic pendulum

设摆长 `L`，摆角 `theta_t`。

```text
E_t = 0.5 * L^2 * theta_dot(t)^2 + gL(1 - cos(theta_t))
L_t = ||bob_t - pivot||
T = mean_zero_crossing_period(theta)
```

invariants：

- energy
- pendulum length
- period

#### Non-holonomic pendulum

只保留：

- approximate energy

#### Double pendulum

第一版 benchmark 不强制 total energy 精确实现，默认只做：

- link length 1 consistency
- link length 2 consistency
- trajectory boundedness

等第二版再补全双摆总能量公式。

### 6.13.3 时间稳定性评分

对某个 invariant 序列 `q_t`：

先计算均值和标准差：

```text
mu_q = mean_t q_t
sigma_q = std_t q_t
```

如果 `|mu_q| >= 10 * sigma_q`，使用相对标准差：

```text
rstd_q = sigma_q / (|mu_q| + eps)
S_q = max(1 - rstd_q / k_q, 0)
```

默认 `k_q = 1.0`

如果 `|mu_q| < 10 * sigma_q`，说明均值接近 0，相对标准差不稳定，改用绝对标准差：

```text
S_q = max(1 - sigma_q / tau_q, 0)
```

其中 `tau_q` 从 real reference subset 的 95% 分位数估计。

### 6.13.4 Sliding window

因为 bounce / impact 邻域会天然出现局部剧烈波动，不应该让整个视频直接掉到 0。

但不能像 Morpheus 一样只取 `max window score`，否则模型可以通过“长时间不动 + 一小段正常”刷分。

这里建议采用：

1. window 长度 `w = round(0.2 T)`，约为总长的 20%
2. 全视频滑窗
3. 得到每个 window 的 invariant score
4. 取 `top 30% windows` 的平均，而不是最大值

即：

```text
S_q_final = mean(top_k(window_scores_q, ceil(0.3 * W)))
```

### 6.13.5 Physical Invariance 总分

对一个 sample 的所有 invariant：

```text
S_inv = mean_q S_q_final
```

如果 sample 被 discard，则：

```text
S_inv = 0
```

---

## 6.14 Causal / Counterfactual Consistency

### 6.14.1 适用场景

这个部分是 world model benchmark 里很关键但大多数 benchmark 没做细的地方。

适用于：

- 风力强弱改变 -> 轨迹 / 烟雾偏移是否改变
- 推左 / 推右 -> 对象是否往对应方向运动
- 光源方向改变 -> 阴影 / 高光是否对应变化
- 施力大小改变 -> 位移 / 速度 / 撞击结果是否单调变化

### 6.14.2 数据组织

counterfactual task 必须按 group 成组存储：

```json
{
  "counterfactual_group": "wind_balloon_001",
  "intervention_name": "wind_strength",
  "intervention_value": 0.8,
  "expected_effect": {
    "observable": "plume_centroid_x",
    "direction": "increase"
  }
}
```

同一 group 至少 3 个 intervention level。

### 6.14.3 Observable 提取

不要依赖 VLM 主观判断，统一转成可测 observable。

#### 流体 / 烟雾

- ROI：由标注文件提供
- 先在 ROI 内做 mask 提取
- 计算 plume centroid、spread、dominant flow direction

```text
centroid_x = sum_p x * mask(p) / sum_p mask(p)
spread = sqrt(sum_p ||p - centroid||^2 * mask(p) / sum_p mask(p))
```

#### 光照 / 阴影 / 反射

- 目标物体 / 地面 / 反射区域 ROI 由标注文件给出
- 计算亮度重心、亮度分布 KL、高亮区域位置

```text
I_luma = 0.299 R + 0.587 G + 0.114 B
highlight_centroid = sum_p p * 1[I_luma(p) > tau] / sum_p 1[I_luma(p) > tau]
```

#### 热学

RGB 数据里不直接测温，第一版只测**热现象可见后果**：

- 蒸汽 / 热流 plume 高度
- 受热物体颜色变化
- 扩散范围

### 6.14.4 Causal Direction Accuracy

对两条 intervention `a < b`，observable 分别为 `o_a, o_b`。

若预期 effect 是 increase：

```text
CDA(a, b) = 1[o_b - o_a > delta]
```

若预期 effect 是 decrease：

```text
CDA(a, b) = 1[o_b - o_a < -delta]
```

默认 `delta = 0.02 * observable_range_ref`

总体：

```text
S_cda = mean_all_pairs CDA(a, b)
```

### 6.14.5 Monotonic Consistency

对 intervention 值序列 `u_i` 和 observable `o_i`，计算 Spearman rank correlation：

```text
rho = Spearman(u, o)
S_mono = (rho + 1) / 2
```

如果预期是 decreasing，则对 `-o` 计算。

### 6.14.6 Effect Size Separation

```text
ESS = |mean(o_high) - mean(o_low)| / (std_pooled + eps)
S_ess = min(ESS / tau_ess, 1)
```

默认 `tau_ess = 2.0`

### 6.14.7 Counterfactual 总分

```text
S_causal =
  0.40 * S_cda +
  0.35 * S_mono +
  0.25 * S_ess
```

---

## 6.15 Fluid / Optics / Thermodynamics 扩展 Track

对于这三类非刚体 / 非经典刚体任务，建议优先使用**Physics-IQ 风格**的 mask-based metric，而不是强行套解析 ODE。

### 6.15.1 为什么

因为这类现象通常没有简单单物体轨迹：

- 流体会变形
- 反射 / 阴影本质上是光照几何
- 热现象往往体现为区域变化，而不是单点运动

### 6.15.2 需要的数据

每个 scenario 建议至少有：

- `take-1` 和 `take-2` 两个真实物理重复实验
- 每个 take 的可评估区域 binary mask 视频
- 最好 2~3 个视角

### 6.15.3 单样本 metric

#### A. Spatiotemporal IoU

对 reference mask `M_ref_t` 与 generated mask `M_gen_t`：

```text
IoU_t = |M_ref_t ∩ M_gen_t| / |M_ref_t ∪ M_gen_t|
S_stiou = mean_t IoU_t
```

#### B. Spatial IoU

先把整个视频时间维做 OR：

```text
M_ref_spatial = OR_t M_ref_t
M_gen_spatial = OR_t M_gen_t
S_siou = IoU(M_ref_spatial, M_gen_spatial)
```

#### C. Weighted Spatial IoU

对像素位置统计其在时间上的占用概率：

```text
W_ref(p) = (1 / T) * sum_t M_ref_t(p)
W_gen(p) = (1 / T) * sum_t M_gen_t(p)
```

然后：

```text
S_wsiou = sum_p min(W_ref(p), W_gen(p)) / sum_p max(W_ref(p), W_gen(p))
```

#### D. RGB MSE

```text
MSE_rgb = mean_t mean_p ||I_ref_t(p) - I_gen_t(p)||_2^2
S_rgb = 1 - norm(MSE_rgb)
```

### 6.15.4 Physical Variance Calibration

如果有 `take-1` 和 `take-2` 两个真实重复实验，用它们之间的差异当作“可接受自然波动”：

```text
Var_phys = mean(
  variance_spatiotemporal_iou,
  variance_spatial_iou,
  variance_weighted_spatial_iou,
  - variance_mse
)
```

然后把 generated sample 的 `S_stiou / S_siou / S_wsiou / S_rgb` 按真实方差归一化。

这个思路直接来自 Physics-IQ，适合 fluid / optics / thermodynamics。

### 6.15.5 非刚体 physics 分数

```text
S_nonrigid =
  0.35 * S_stiou +
  0.20 * S_siou +
  0.25 * S_wsiou +
  0.20 * S_rgb
```

如果有真实重复实验的 variance calibration，则优先报告 calibrated 版本。

---

## 6.16 Rollout Quality Decay

### 6.16.1 目的

只报最终帧质量不够。需要看 rollout 越长，质量衰减多快。

### 6.16.2 时间桶

把视频按时间切 4 个 horizon：

- `H1 = [0, 0.25T)`
- `H2 = [0.25T, 0.5T)`
- `H3 = [0.5T, 0.75T)`
- `H4 = [0.75T, T]`

### 6.16.3 每个 horizon 统计

对每个 `Hk` 计算：

- `CLIP-IQA+`
- `Aesthetic`
- `Photometric Consistency`
- 可选：`LPIPS` / `SSIM`（如果有 reference）

得到综合质量：

```text
Q_k =
  0.35 * norm(CLIP-IQA+_k) +
  0.20 * norm(Aesthetic_k) +
  0.25 * S_photo_k +
  0.20 * S_style_k
```

### 6.16.4 衰减分数

```text
S_rollout_auc = mean_k Q_k
S_rollout_drop = max(Q_1 - Q_4, 0)
S_rollout = 0.7 * S_rollout_auc + 0.3 * (1 - S_rollout_drop)
```

---

## 6.17 Interactive / Real-time Track

### 6.17.1 只对在线模型启用

不要把 offline model 和 online model 混着比。

### 6.17.2 记录指标

1. `Latency p50 / p90`
2. `Step FPS`
3. `Online camera control`
4. `Online object response`
5. `Online drift`

### 6.17.3 Latency

单步 latency：

```text
L_t = t_frame_out - t_action_in
```

分数：

```text
S_latency = max(1 - (p90_latency - budget_ms) / budget_ms, 0)
```

默认 `budget_ms = 200`

### 6.17.4 Online drift

在连续 `N = 16` 步交互下，每一步都计算相机 / 对象控制误差。

```text
drift = mean_t error_t - error_1
S_drift = 1 - norm(max(drift, 0))
```

### 6.17.5 Interactive score

```text
S_interactive =
  0.30 * S_latency +
  0.25 * S_cam_online +
  0.25 * S_obj_online +
  0.20 * S_drift
```

---

## 6.18 Dataset-level Distribution Metrics

这一节是**数据集级**指标，不和 per-sample metric 混在一起。单样本上不要算 FVD。

### 6.18.1 Fréchet Video Distance (FVD)

适用于：

- 模型整体分布是否接近真实 benchmark 视频分布
- 按 domain / horizon bucket / physics subset 做整体对比

步骤：

1. 从 generated videos 和 real reference videos 中各采样一批视频
2. 用预训练 I3D encoder 提取视频 embedding
3. 对 generated / real 两组 embedding 计算均值和协方差

设：

- `mu_g, Sigma_g` 为 generated embedding 的均值和协方差
- `mu_r, Sigma_r` 为 real embedding 的均值和协方差

公式：

```text
FVD =
  ||mu_g - mu_r||_2^2 +
  Tr(Sigma_g + Sigma_r - 2 * (Sigma_g Sigma_r)^(1/2))
```

分数：

```text
S_fvd = 1 - norm(FVD)
```

建议同时报告：

- `FVD_all`
- `FVD_indoor`
- `FVD_outdoor`
- `FVD_physics`
- `FVD_long_horizon`

### 6.18.2 Video-Text Alignment

如果 benchmark 中每个 sample 都有 target prompt，则可以补一个 dataset-level 语义一致性：

1. 对每个视频抽 anchor frames
2. 用 CLIP image encoder 编码 anchor
3. 用 CLIP text encoder 编码 target prompt
4. 对每个视频求 mean cosine similarity

```text
VTAlign = mean_samples mean_anchor cosine(E_img(I_anchor), E_txt(prompt))
```

### 6.18.3 Video-Video Retrieval Consistency

如果有 reference video，可把 generated video 与 reference video 编码到同一 video embedding 空间：

```text
VVSim = mean_samples cosine(E_video(V_gen), E_video(V_ref))
```

推荐只把它作为补充指标，不计入总分。

---

## 6.19 Autonomous Driving Optional Track

这个 track 只对 driving subset 启用，参考 DrivingGen 的设计。

### 6.19.1 需要的额外 annotation

- `ego_trajectory_input`
- `ego_trajectory_reference`
- 可选 `agent_boxes` / `agent_masks`
- domain tags 中明确：
  - weather
  - time of day
  - region
  - maneuver type

### 6.19.2 Fréchet Trajectory Distance (FTD)

如果 benchmark 中能从视频或标注里抽出 ego trajectory，则定义 trajectory embedding 或直接使用 resampled trajectory points。

推荐第一版直接用 resampled trajectory vector：

```text
tau = [x_1, y_1, x_2, y_2, ..., x_T, y_T]
```

然后对 generated / real 两组 trajectory vector 计算：

```text
FTD =
  ||mu_tau_g - mu_tau_r||_2^2 +
  Tr(Sigma_tau_g + Sigma_tau_r - 2 * (Sigma_tau_g Sigma_tau_r)^(1/2))
```

### 6.19.3 Average Displacement Error (ADE)

若有输入 ego trajectory `tau_in` 和生成后估计出的 ego trajectory `tau_gen`：

```text
ADE = (1 / T) * sum_t ||tau_gen(t) - tau_in(t)||_2
S_ade = 1 - norm(ADE)
```

### 6.19.4 Dynamic Time Warping (DTW)

当生成速度分布与输入速度不完全同步，但总体轨迹形状仍有可比性时，补充 DTW：

```text
DTW(tau_gen, tau_in) = min_path sum_(i,j in path) ||tau_gen(i) - tau_in(j)||_2
S_dtw = 1 - norm(DTW)
```

### 6.19.5 Agent Consistency / Missing

对于前向驾驶视频，还应补两个 agent-level 指标：

#### agent consistency

对已跟踪 agent 的轨迹连续性：

```text
S_agent_cons =
  1 - norm(mean_agent mean_t ||p_t - p_hat_t(local_smooth)||_2)
```

其中 `p_hat_t(local_smooth)` 是局部平滑轨迹。

#### agent missing

统计前景 agent 在无合理遮挡条件下突然消失的比例：

```text
MissingRate = (# disappearing agents) / (# tracked agents)
S_agent_missing = 1 - MissingRate
```

### 6.19.6 Driving score

```text
S_driving =
  0.20 * S_fvd +
  0.20 * S_ftd +
  0.20 * S_ade +
  0.15 * S_dtw +
  0.15 * S_agent_cons +
  0.10 * S_agent_missing
```

这个分数默认只在 driving subset 报，不并入全局 `S_worldarena`，避免 benchmark 被某个单一 domain 主导。

---

## 7. Overall Score 聚合

### 7.1 Core score

对于**当前 WorldArena 数据驱动的 official**，总分默认采用“四个 formal suite + auxiliary + experimental”的结构：

- `S_image_static`
- `S_image_dynamic`
- `S_video_static`
- `S_video_dynamic`
- `S_auxiliary`
- `S_experimental`

其中只有前四者进入 official 总分。

#### 7.1.1 `S_image_static`

```text
S_image_static =
  0.50 * S_scene_taxonomy_static +
  0.25 * S_style_static +
  0.25 * S_quality_image_static
```

#### 7.1.2 `S_image_dynamic`

```text
S_image_dynamic =
  0.50 * S_dynamic_category_image +
  0.25 * S_style_dynamic_image +
  0.25 * S_quality_image_dynamic
```

#### 7.1.3 `S_video_static`

```text
S_video_static =
  0.30 * S_scene_taxonomy_video_static +
  0.15 * S_style_video_static +
  0.20 * S_subjective_video_static +
  0.20 * S_rollout_video_static +
  0.15 * S_motion_smoothness_static
```

#### 7.1.4 `S_video_dynamic`

```text
S_video_dynamic =
  0.30 * S_dynamic_category_video +
  0.10 * S_style_video_dynamic +
  0.20 * S_subjective_video_dynamic +
  0.15 * S_rollout_video_dynamic +
  0.15 * S_motion_magnitude +
  0.10 * S_motion_smoothness_dynamic
```

#### 7.1.5 Official 总分

```text
S_worldarena =
  0.15 * S_image_static +
  0.15 * S_image_dynamic +
  0.30 * S_video_static +
  0.40 * S_video_dynamic
```

说明：

- `camera_control` / `object_control` / `geom_consistency` / `long_delayed` / `physics`
  在当前数据形态下默认不进入 `S_worldarena`
- 这些能力若有 sidecar 支持，只进入 subset report 或 `S_experimental`

### 7.2 Auxiliary score

```text
S_auxiliary =
  mean(S_nonformal_video, S_nonformal_image)
```

`S_auxiliary` 默认不并入 `S_worldarena`。

### 7.3 Experimental score

实验轨道单独报告：

```text
S_experimental =
  mean(
    S_camera_control_subset,
    S_object_control_subset,
    S_geom_subset,
    S_long_delayed_exp,
    S_physics_exp,
    S_masked_variant_exp
  )
```

`S_experimental` 默认不并入 `S_worldarena`。

### 7.4 不要只报单一总分

leaderboard 至少同时报：

- `S_worldarena`
- `S_image_static`
- `S_image_dynamic`
- `S_video_static`
- `S_video_dynamic`
- `S_auxiliary`
- `S_experimental`
- `S_quality_image_static`
- `S_quality_image_dynamic`
- `S_subjective_video_static`
- `S_subjective_video_dynamic`
- `DiscardRate`

## 8. Domain Slice 与报表要求

以下维度在**字段存在时**必须做分组统计；字段不存在时要显式标成 `unknown`，不要隐式补全：

- formal / non-formal
- image / video
- static / dynamic
- indoor / outdoor
- photorealistic / stylized
- image_static / image_dynamic / video_static / video_dynamic / video_nonformal
- static leaf category，例如 `living_spaces` / `city` / `verdant_landscape`
- articulated / deformable / fluid / multi_motion / rigid
- AAA_Games（仅 auxiliary）

每个 metric 需要输出：

- mean
- median
- std
- 95% bootstrap CI
- num_samples

bootstrap：

- `n_bootstrap = 1000`
- 按 sample 级别重采样

## 9. 实现模块建议

建议新建：

```text
src/worldarena/evaluation/worldmodel_benchmark/
  __init__.py
  schemas.py
  loader.py
  runner.py
  aggregate.py
  cache.py
  report.py
  inventory.py
  suites.py
  data_sources/
    formal_static_image.py
    formal_dynamic_image.py
    formal_static_video.py
    formal_dynamic_video.py
    nonformal_video.py
    sidecar_scanner.py
  metrics/
    camera_control.py
    object_control.py
    scene_alignment.py
    long_revisit.py
    long_delayed.py
    met3r_wrapper.py
    reprojection.py
    photometric.py
    style.py
    quality.py
    rollout.py
    interactive.py
    physics/
      tracking.py
      preprocessing.py
      discard.py
      dynamical_fit.py
      pinn_fit.py
      invariance.py
      causal.py
      nonrigid.py
```

### 9.1 统一 metric 接口

```python
class Metric:
    name: str
    requires: list[str]

    def prepare(self, sample, cache) -> None:
        ...

    def compute(self, sample, cache) -> dict:
        """
        return {
            "raw": ...,
            "normalized": ...,
            "artifacts": {...}
        }
        """
```

### 9.2 缓存依赖

例如：

- `camera_control` 依赖 `camera_pose_est`
- `photometric` 依赖 `optical_flow`
- `physics` 依赖 `sam2_tracks`, `depth_anything_depth`
- `long_revisit` 依赖 `met3r_pairs`, `groundingdino_detections`

## 10. MVP 实现顺序

第一版不要贪多，建议按下面顺序做：

### Phase 1

- `build_worldarena_inventory.py`
- `formal_static_image` loader / schema 校验
- `formal_dynamic_image` loader / schema 校验
- `formal_static_video` loader / schema 校验
- `formal_dynamic_video` loader / schema 校验
- `category_path -> tags` 解析器
- frame decode
- CLIPScore / scene alignment
- dynamic-category alignment
- MEt3R wrapper
- CLIP-IQA+ / aesthetic
- VGG Gram matrix
- motion magnitude
- motion smoothness

### Phase 2

- rollout quality
- dataset-level FVD / suite aggregate
- image-static / image-dynamic / video-static / video-dynamic suite 报表
- official aggregate + leaderboard

### Phase 3

- sidecar scanner
- masked variant（仅当 mask 存在）
- subset-level camera / object / geometry metrics
- delayed-effect experimental
- analytic dynamical fit
- physical invariance experimental
- `nonformal_video` experimental loader
- causal / counterfactual（future-expansion 开关）
- interactive benchmark（future-expansion 开关）

## 11. Smoke Test 要求

每个新增 metric 至少配一个 deterministic smoke test。

推荐：

1. `test_worldarena_inventory_smoke.py`
2. `test_formal_static_image_loader_smoke.py`
3. `test_formal_dynamic_image_loader_smoke.py`
4. `test_formal_static_video_loader_smoke.py`
5. `test_formal_dynamic_video_loader_smoke.py`
6. `test_scene_alignment_smoke.py`
7. `test_dynamic_category_smoke.py`
8. `test_motion_smoothness_smoke.py`
9. `test_rollout_quality_smoke.py`
10. `test_dynamical_projectile_smoke.py`
11. `test_invariance_energy_smoke.py`
12. `test_counterfactual_monotonic_smoke.py`

## 12. 需要明确写死到 config 的超参数

不能散落在代码里，统一写入 `configs/worldmodel_benchmark.yaml`：

- target FPS
- anchor sampling strategy
- DROID image size / stride
- GroundingDINO thresholds
- SAM2 checkpoint
- SEA-RAFT checkpoint
- MEt3R backbone / feature backbone / upsampler
- Savitzky-Golay window / order
- discard thresholds
- normalization bounds
- aggregate weights
- latency budget

## 13. 最终输出文件

每次 run 至少输出：

```text
outputs/{run_id}/
  run_manifest.json
  per_sample_scores.jsonl
  per_metric_summary.json
  domain_breakdown.json
  leaderboard.csv
  plots/
    radar_core.png
    long_rollout_decay.png
    physics_discard_breakdown.png
    domain_heatmap.png
```

每个 sample 的结果建议长这样：

```json
{
  "sample_id": "video_dynamic_photorealistic_articulated_0001",
  "track": "scene_control",
  "metrics": {
    "clip_score": {"raw": 0.84, "normalized": 0.84},
    "dynamic_category": {"raw": 0.88, "normalized": 0.88},
    "motion_magnitude": {"raw": 0.71, "normalized": 0.71},
    "motion_smoothness": {"raw": 0.79, "normalized": 0.79}
  },
  "track_score": 0.81
}
```

## 14. 推荐参考实现与论文对应关系

下列对应关系建议在文档和代码注释里写清楚：

- `camera_error` / `object_detection` / `clip_score` / `gram_matrix` / `optical_flow_aepe` / `motion_smoothness`：参考 **WorldScore**
- `met3r`：参考 **MEt3R**
- `discard rate` / `dynamical score` / `physical invariance` / `SAM2 + Depth Anything V2 trajectory extraction`：参考 **Morpheus**
- `fluid / optics / thermodynamics` 的 mask-based 多视角重复实验评测：参考 **Physics-IQ**
- `driving domain` 的 open-domain vs ego-conditioned reporting、trajectory alignment 额外指标：参考 **DrivingGen**

## 15. 结论：第一版 benchmark 的默认主张

第一版 benchmark 建议遵循下面 10 条约束：

1. **先基于现有 WorldArena formal 数据把 official 跑通，再谈外部扩展。**
2. **official 只使用 `is_formal = true` 的资产。**
3. **non-formal 数据只进入 auxiliary / experimental。**
4. **不要默认假设 pose、instruction、mask、physics sidecar 一定存在。**
5. **reference-grounded 和 physics-grounded 分开实现。**
6. **geometry 与 photometric 分开，不要混成一个分数。**
7. **physics 默认先用解析拟合，不先上 PINN。**
8. **所有总分都要能追溯到 per-sample raw metric。**
9. **experimental 赛道不并入 official 主总分。**
10. **所有实际样本数量由 freeze manifest 自动枚举，不写死在代码里。**

如果后续 coding agent 按这份规格实现，第一版就已经足够支撑：

- formal static / dynamic image benchmark
- formal static / dynamic video benchmark
- scene / style / dynamic-taxonomy alignment
- quality / motion / rollout degradation
- sidecar-activated experimental 子榜

## 16. Reference Links

- [WorldScore](https://arxiv.org/abs/2504.00983)
- [WorldScore project page](https://haoyi-duan.github.io/WorldScore/)
- [WorldScore official code](https://github.com/haoyi-duan/WorldScore)
- [MEt3R](https://arxiv.org/abs/2501.06336)
- [MEt3R project page](https://geometric-rl.mpi-inf.mpg.de/met3r/)
- [MEt3R code](https://github.com/mohammadasim98/met3r)
- [Morpheus](https://arxiv.org/abs/2504.02918)
- [Morpheus project page](https://physics-from-video.github.io/morpheus-bench/)
- [Physics-IQ benchmark](https://physics-iq.github.io/)
- [Physics-IQ code](https://github.com/google-deepmind/physics-IQ-benchmark)
- [DrivingGen](https://arxiv.org/abs/2601.01528)
- [DrivingGen project page](https://drivinggen-bench.github.io/)
- [CameraBench](https://arxiv.org/abs/2504.15376)
- [CameraBench project page](https://linzhiqiu.github.io/papers/camerabench/)
- [VBench](https://github.com/Vchitect/VBench)
- [WorldArena dataset tree](https://huggingface.co/datasets/OpenEnvisionLab/WorldArena/tree/main)
- [Pexels license](https://www.pexels.com/license/)
- [Pexels API](https://www.pexels.com/api/)
- [Ego4D docs](https://ego4d-data.org/docs/)
- [Ego4D start here](https://ego4d-data.org/docs/start-here/)
- [Ego-Exo4D docs](https://docs.ego-exo4d-data.org/)
- [Ego-Exo4D download docs](https://docs.ego-exo4d-data.org/download/)
- [BDD100K paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.pdf)
- [BDD100K official repo](https://github.com/bdd100k/bdd100k)
- [Argoverse 2](https://www.argoverse.org/av2.html)
- [Argoverse 2 Sensor docs](https://argoverse.github.io/user-guide/datasets/sensor.html)
- [Argoverse terms](https://argoverse.github.io/user-guide/terms_and_conditions.html)
- [Waymo Motion](https://www.waymo.com/open/data/motion/)
- [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php)
- [nuScenes](https://motional.com/nuscenes)
- [BridgeData V2](https://bridgedata-v2.github.io/)
- [BridgeData V2 paper](https://arxiv.org/abs/2308.12952)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [Open X-Embodiment paper](https://autolab.berkeley.edu/assets/publications/media/Open_X_Embodiment__Robotic_Learning_Datasets_and_RT_X_Models%20%281%29.pdf)
- [Matterport HM3D access](https://matterport.com/partners/meta)
- [CARLA](https://carla.org/)
- [CARLA docs](https://carla.readthedocs.io/)
- [Kubric docs](https://kubric.readthedocs.io/en/latest/)
