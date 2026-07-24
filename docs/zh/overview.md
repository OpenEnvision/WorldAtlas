# 数据概览

!!! info "生成内容"
    当你新增、删除或重组源数据资产后，请运行 `worldarena build-docs --config config/datasets.yaml`。

<p class="doc-provenance">快照说明：由于最近一次 `worldarena build-docs` 运行时本地数据根目录不可用，下列面板与图表使用本 checkout 中缓存的 `docs/generated/` 产物。当本地数据根目录可用时，请重新运行该命令以进行实时扫描。</p>

## 概述

<p class="overview-lede">WorldAtlas Arena 将多任务基准语料、六维评估契约与世界模型适配器注册表整合在同一工作区。下列面板从媒体规模、任务范式、指标面与模型覆盖等多个角度汇总统计，再进入集合明细与样例画廊。</p>

## 基准规模

<p class="overview-section-lede">总量指标、媒体库存、任务范式与基准契约面。</p>

<div class="overview-page">

<div class="overview-section">
  <p class="overview-section-label">总量指标</p>
  <p class="overview-section-note">基准条目、套件、指标与模型覆盖的主统计。</p>
<div class="overview-hero-grid">
<div class="overview-hero-card overview-hero-card--primary">
  <div class="overview-hero-label">基准条目数</div>
  <div class="overview-hero-value">11,653</div>
  <div class="overview-hero-note">图像、视频与 physics 套件的主基准行</div>
</div>
<div class="overview-hero-card overview-hero-card--green">
  <div class="overview-hero-label">正式评测套件</div>
  <div class="overview-hero-value">4</div>
  <div class="overview-hero-note">公开基准契约，定义于 `config/benchmark.yaml`</div>
</div>
<div class="overview-hero-card overview-hero-card--warm">
  <div class="overview-hero-label">文档化指标</div>
  <div class="overview-hero-value">47</div>
  <div class="overview-hero-note">基准文档中的公开指标清单</div>
</div>
<div class="overview-hero-card overview-hero-card--blue">
  <div class="overview-hero-label">评估维度</div>
  <div class="overview-hero-value">6</div>
  <div class="overview-hero-note">聚合写入 `per_dimension_summary.json` 的六个主轴</div>
</div>
<div class="overview-hero-card overview-hero-card--ink">
  <div class="overview-hero-label">已注册模型</div>
  <div class="overview-hero-value">25</div>
  <div class="overview-hero-note">`config/models/` 下的适配器配置</div>
</div>
<div class="overview-hero-card overview-hero-card--green">
  <div class="overview-hero-label">已标注 Rows</div>
  <div class="overview-hero-value">5,496</div>
  <div class="overview-hero-note">47.2% 的基准条目附带监督侧车数据</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">媒体库存</p>
  <p class="overview-section-note">基准语料中的图像、视频与 physics 资产。</p>
<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">图像</div>
  <div class="metric-value">5,000</div>
  <div class="metric-note">静态与动态图像资产</div>
</div>
<div class="metric-card">
  <div class="metric-label">视频</div>
  <div class="metric-value">5,253</div>
  <div class="metric-note">静态、动态与 gameplay 视频</div>
</div>
<div class="metric-card">
  <div class="metric-label">物理</div>
  <div class="metric-value">1,400</div>
  <div class="metric-note">物理导向的视频资产</div>
</div>
<div class="metric-card">
  <div class="metric-label">带 Mask 的动态样本</div>
  <div class="metric-value">2,124</div>
  <div class="metric-note">含分割 mask 的动态图像与 gameplay 条目</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">任务范式</p>
  <p class="overview-section-note">基准条目在 static、dynamic、interaction 与 physics 轨道上的分布。</p>
<div class="overview-regime-grid">
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">静态世界生成</span>
    <span class="overview-regime-share">63.3%</span>
  </div>
  <div class="overview-regime-value">7,372</div>
  <div class="overview-regime-note">相机或布局变化下的场景一致性扩展</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">动态世界生成</span>
    <span class="overview-regime-share">20.7%</span>
  </div>
  <div class="overview-regime-value">2,416</div>
  <div class="overview-regime-note">固定或受控视角下的场景内时序演化</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">交互条件生成</span>
    <span class="overview-regime-share">4.0%</span>
  </div>
  <div class="overview-regime-value">465</div>
  <div class="overview-regime-note">Gameplay 与动作条件 rollout</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">物理指标</span>
    <span class="overview-regime-share">12.0%</span>
  </div>
  <div class="overview-regime-value">1,400</div>
  <div class="overview-regime-note">参考运动与仿真器支持的物理评测轨道</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">评估框架</p>
  <p class="overview-section-note">正式套件区分静态、动态与交互条件生成，并共享统一的报告契约；各套件在 `config/benchmark.yaml` 中声明适用的指标向量。</p>

<table>
<thead>
<tr><th align="left">suite</th><th align="left">modality</th><th align="left">regime</th><th align="right">metrics</th></tr>
</thead>
<tbody>
<tr><td align="left">image_static</td><td align="left">image → video</td><td align="left">static</td><td align="right">43</td></tr>
<tr><td align="left">image_dynamic</td><td align="left">image → video</td><td align="left">dynamic</td><td align="right">33</td></tr>
<tr><td align="left">video_static</td><td align="left">video → video</td><td align="left">static</td><td align="right">37</td></tr>
<tr><td align="left">video_dynamic</td><td align="left">video → video</td><td align="left">dynamic</td><td align="right">43</td></tr>
</tbody>
</table>

  <p class="overview-section-label">按维度划分的指标</p>

<div class="overview-dimension-bars">
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">长序列</span>
    <span class="overview-dimension-count">1 项指标 · 0 项官方</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--long_sequence" style="width: 8%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">动作控制</span>
    <span class="overview-dimension-count">12 项指标 · 4 项官方</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--action_control" style="width: 86%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">3D/4D 一致性</span>
    <span class="overview-dimension-count">7 项指标 · 2 项官方</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--consistency_3d_4d" style="width: 50%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">物理</span>
    <span class="overview-dimension-count">11 项指标 · 10 项官方</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--physics" style="width: 79%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">质量</span>
    <span class="overview-dimension-count">14 项指标 · 5 项官方</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--quality" style="width: 100%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">实时性</span>
    <span class="overview-dimension-count">2 项指标 · 2 项官方</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--real_time" style="width: 14%;"></div>
  </div>
</div>
</div>

  <p class="overview-section-label">官方指标 vs 补充指标</p>

<div class="overview-split-grid">
<div class="overview-split-card overview-split-card--official">
  <div class="overview-split-label">官方指标面</div>
  <div class="overview-split-value">23</div>
  <div class="overview-split-note">公开清单中进入 headline 或条件性 official 聚合的指标</div>
</div>
<div class="overview-split-card overview-split-card--supplementary">
  <div class="overview-split-label">补充指标</div>
  <div class="overview-split-value">24</div>
  <div class="overview-split-note">补充几何、重建与长序列补充指标</div>
</div>
<div class="overview-split-card overview-split-card--experimental">
  <div class="overview-split-label">实验套件</div>
  <div class="overview-split-value">19</div>
  <div class="overview-split-note">在 `config/benchmark.yaml` 中单独声明、不计入 headline 的切片</div>
</div>
</div>

  <p class="overview-section-note">指标适用性、归一化与聚合语义见 <a href="benchmark/metrics/index.md">指标概览</a>。</p>

  <div class="overview-quick-links">
    <a href="benchmark/metrics/index.md">指标概览</a>
    <a href="benchmark/index.md">基准评测总览</a>
    <a href="development.md">开发指南</a>
  </div>
</div>

</div>

## 拓扑

```mermaid
graph TD
  root[WorldAtlas Arena]
  root --> images[Image]
  root --> videos[Video]
  root --> physics[Physics]
  images --> static_images[静态图像 • 3372]
  images --> dynamic_images[动态图像 • 1628]
  videos --> static_videos[静态视频 • 4000]
  videos --> dynamic_videos[动态视频 • 788]
  videos --> aaa_games[AAA Games • 465]
  physics --> physics_videos[Physics 视频 • 1400]
```

## 集合汇总

| group     | collection     | modality   |   items |   annotated |   masked |
|:----------|:---------------|:-----------|--------:|------------:|---------:|
| benchmark | dynamic_images | image      |    1628 |        1628 |     1628 |
| benchmark | static_images  | image      |    3372 |        3372 |        0 |
| benchmark | physics_videos | physics    |    1400 |           0 |        0 |
| benchmark | aaa_games      | video      |     465 |         465 |      465 |
| benchmark | dynamic_videos | video      |     788 |           0 |        0 |
| benchmark | static_videos  | video      |    4000 |          31 |       31 |

## 集合规模分布

<iframe class="chart-frame" src="../generated/charts/overview-collection-volume.html" title="集合规模分布" loading="lazy"></iframe>

## 基准分类层级

<iframe class="chart-frame" src="../generated/charts/overview-benchmark-hierarchy.html" title="基准分类层级" loading="lazy"></iframe>

## 图像样例

<div class="sample-grid sample-grid--image">
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/01-dynamic_images-003.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/003.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/02-static_images-000.jpg" alt="Static / Photorealistic / Dining Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Dining Spaces</strong>
    <span>photorealistic/indoor/dining_spaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/03-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Deformable</strong>
    <span>photorealistic/images/deformable/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/04-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Fluid</strong>
    <span>photorealistic/images/fluid/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/05-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Multi Motion</strong>
    <span>photorealistic/images/multi_motion/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/06-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Rigid</strong>
    <span>photorealistic/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/07-dynamic_images-000.jpg" alt="Dynamic / Stylized / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Articulated</strong>
    <span>stylized/images/articulated/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/08-dynamic_images-001.jpg" alt="Dynamic / Stylized / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Deformable</strong>
    <span>stylized/images/deformable/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/09-dynamic_images-000.jpg" alt="Dynamic / Stylized / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Fluid</strong>
    <span>stylized/images/fluid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/10-dynamic_images-000.jpg" alt="Dynamic / Stylized / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Multi Motion</strong>
    <span>stylized/images/multi_motion/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/11-dynamic_images-000.jpg" alt="Dynamic / Stylized / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Rigid</strong>
    <span>stylized/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/12-static_images-0000.jpg" alt="Static / Photorealistic / Living Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Living Spaces</strong>
    <span>photorealistic/indoor/living_spaces/0000.png</span>
  </figcaption>
</figure>
</div>



## Physics 样例

<div class="sample-grid sample-grid--physics">
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/posters/01-physics_videos-00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.jpg" alt="Basic Physics Consistency / Acceleration" loading="lazy">
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/posters/15-physics_videos-06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.jpg" alt="Basic Physics Consistency / Uniform Motion" loading="lazy">
  <figcaption>
    <strong>Basic Physics Consistency / Uniform Motion</strong>
    <span>videos_by_category/basic_physics_consistency/uniform_motion/06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.mp4</span>
  </figcaption>
</figure>
</div>
