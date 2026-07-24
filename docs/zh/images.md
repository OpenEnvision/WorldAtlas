# 图像统计

!!! info "生成内容"
    当你新增、删除或重组源数据资产后，请运行 `worldarena build-docs --config config/datasets.yaml`。

## 概述

<p class="overview-lede">图像语料覆盖静态场景帧与带运动标签的动态静帧，分别服务于 `image_static` 与 `image_dynamic` 基准套件。下列面板从库存规模、分类覆盖、mask 监督与适用指标等角度汇总统计，再进入明细表与交互图表。</p>

## 集合规模

<p class="overview-section-lede">总量指标, taxonomy breadth, and static/dynamic regime mix for the image corpus.</p>

<div class="overview-page">

<div class="overview-section">
  <p class="overview-section-label">总量指标</p>
  <p class="overview-section-note">Primary inventory counts and benchmark contract surface.</p>
<div class="overview-hero-grid">
<div class="overview-hero-card overview-hero-card--primary">
  <div class="overview-hero-label">图像条目</div>
  <div class="overview-hero-value">5,000</div>
  <div class="overview-hero-note">全部扫描到的图像基准行</div>
</div>
<div class="overview-hero-card overview-hero-card--blue">
  <div class="overview-hero-label">Mask 覆盖</div>
  <div class="overview-hero-value">1,628</div>
  <div class="overview-hero-note">32.6% 的图像条目附带分割 mask</div>
</div>
<div class="overview-hero-card overview-hero-card--ink">
  <div class="overview-hero-label">正式图像套件</div>
  <div class="overview-hero-value">2</div>
  <div class="overview-hero-note">公开契约：`image_static`、`image_dynamic`</div>
</div>
<div class="overview-hero-card overview-hero-card--green">
  <div class="overview-hero-label">适用图像指标</div>
  <div class="overview-hero-value">46</div>
  <div class="overview-hero-note">`image_static` 与 `image_dynamic` 的去重指标数</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">分类拓扑 breadth</p>
  <p class="overview-section-note">Distinct style, environment, motion, and resolution labels in inventory.</p>
<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">视觉风格</div>
  <div class="metric-value">2</div>
  <div class="metric-note">写实与风格化静帧</div>
</div>
<div class="metric-card">
  <div class="metric-label">场景环境</div>
  <div class="metric-value">2</div>
  <div class="metric-note">室内与室外场景桶</div>
</div>
<div class="metric-card">
  <div class="metric-label">运动类别</div>
  <div class="metric-value">5</div>
  <div class="metric-note">关节、刚体、流体与复合运动标签</div>
</div>
<div class="metric-card">
  <div class="metric-label">分辨率种类</div>
  <div class="metric-value">693</div>
  <div class="metric-note">库存中的不同分辨率标签</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">Regime mix</p>
  <p class="overview-section-note">How rows split between static layout evaluation and dynamic motion evaluation.</p>
<div class="overview-regime-grid">
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">静态图像</span>
    <span class="overview-regime-share">67.4%</span>
  </div>
  <div class="overview-regime-value">3,372</div>
  <div class="overview-regime-note">场景布局与相机一致性评测</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">动态图像</span>
    <span class="overview-regime-share">32.6%</span>
  </div>
  <div class="overview-regime-value">1,628</div>
  <div class="overview-regime-note">运动、mask 与时序结构指标</div>
</div>
</div>
</div>

</div>

## 基准套件

图像套件在 `config/benchmark.yaml` 中声明适用指标：静态样本侧重布局与相机一致性，动态样本额外覆盖运动与 mask 感知指标。

| suite         | regime   |   items |   metrics |
|:--------------|:---------|--------:|----------:|
| image_static  | static   |    3372 |        43 |
| image_dynamic | dynamic  |    1628 |        33 |

指标适用性与聚合语义见 [指标概览](benchmark/metrics/index.md)。

## 分类拓扑

```mermaid
graph TD
  root[Image Corpus • 5000]
  root --> static[静态图像 • 3372]
  root --> dynamic[动态图像 • 1628]
  static --> indoor[Indoor Scenes]
  static --> outdoor[Outdoor Scenes]
  dynamic --> motion[运动类别]
  dynamic --> masks[Segmentation Masks • 1628]
```

## 静态场景汇总

| style          | environment   | scene                 |   items |
|:---------------|:--------------|:----------------------|--------:|
| photorealistic | indoor        | dining_spaces         |     169 |
| photorealistic | indoor        | living_spaces         |     169 |
| photorealistic | indoor        | public_spaces         |     169 |
| photorealistic | indoor        | workspaces            |     169 |
| stylized       | outdoor       | terrestrial_landscape |     169 |
| photorealistic | outdoor       | aquatic_landscape     |     169 |
| photorealistic | outdoor       | city                  |     169 |
| photorealistic | outdoor       | terrestrial_landscape |     169 |
| stylized       | indoor        | dining_spaces         |     169 |
| photorealistic | outdoor       | verdant_landscape     |     169 |
| stylized       | indoor        | living_spaces         |     169 |
| stylized       | indoor        | public_spaces         |     169 |
| stylized       | outdoor       | aquatic_landscape     |     169 |
| stylized       | indoor        | workspaces            |     169 |
| stylized       | outdoor       | verdant_landscape     |     169 |

## 动态 Mask 汇总

| style          | motion_category   |   items |   masked |   mask_rate |
|:---------------|:------------------|--------:|---------:|------------:|
| photorealistic | articulated       |     169 |      169 |           1 |
| photorealistic | deformable        |     169 |      169 |           1 |
| photorealistic | fluid             |     169 |      169 |           1 |
| photorealistic | multi_motion      |     168 |      168 |           1 |
| photorealistic | rigid             |     108 |      108 |           1 |
| stylized       | articulated       |     169 |      169 |           1 |
| stylized       | deformable        |     169 |      169 |           1 |
| stylized       | fluid             |     169 |      169 |           1 |
| stylized       | multi_motion      |     169 |      169 |           1 |
| stylized       | rigid             |     169 |      169 |           1 |

## 图像样例

<div class="sample-grid sample-grid--image">
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/01-dynamic_images-003.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/003.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/02-static_images-000.jpg" alt="Static / Photorealistic / Dining Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Dining Spaces</strong>
    <span>photorealistic/indoor/dining_spaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/03-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Deformable</strong>
    <span>photorealistic/images/deformable/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/04-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Fluid</strong>
    <span>photorealistic/images/fluid/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/05-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Multi Motion</strong>
    <span>photorealistic/images/multi_motion/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/06-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Rigid</strong>
    <span>photorealistic/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/07-dynamic_images-000.jpg" alt="Dynamic / Stylized / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Articulated</strong>
    <span>stylized/images/articulated/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/08-dynamic_images-001.jpg" alt="Dynamic / Stylized / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Deformable</strong>
    <span>stylized/images/deformable/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/09-dynamic_images-000.jpg" alt="Dynamic / Stylized / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Fluid</strong>
    <span>stylized/images/fluid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/10-dynamic_images-000.jpg" alt="Dynamic / Stylized / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Multi Motion</strong>
    <span>stylized/images/multi_motion/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/11-dynamic_images-000.jpg" alt="Dynamic / Stylized / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Rigid</strong>
    <span>stylized/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/12-static_images-0000.jpg" alt="Static / Photorealistic / Living Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Living Spaces</strong>
    <span>photorealistic/indoor/living_spaces/0000.png</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/13-static_images-001.jpg" alt="Static / Photorealistic / Passageways" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Passageways</strong>
    <span>photorealistic/indoor/passageways/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/14-static_images-002.jpg" alt="Static / Photorealistic / Public Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Public Spaces</strong>
    <span>photorealistic/indoor/public_spaces/002.png</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/15-static_images-003.jpg" alt="Static / Photorealistic / Workspaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Workspaces</strong>
    <span>photorealistic/indoor/workspaces/003.png</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/16-static_images-000.jpg" alt="Static / Photorealistic / Aquatic Landscape" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Aquatic Landscape</strong>
    <span>photorealistic/outdoor/aquatic_landscape/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/17-static_images-003.jpg" alt="Static / Photorealistic / City" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / City</strong>
    <span>photorealistic/outdoor/city/003.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/18-static_images-000.jpg" alt="Static / Photorealistic / Suburb" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Suburb</strong>
    <span>photorealistic/outdoor/suburb/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/19-static_images-006.jpg" alt="Static / Photorealistic / Terrestrial Landscape" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Terrestrial Landscape</strong>
    <span>photorealistic/outdoor/terrestrial_landscape/006.jpeg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/20-static_images-002.jpg" alt="Static / Photorealistic / Verdant Landscape" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Verdant Landscape</strong>
    <span>photorealistic/outdoor/verdant_landscape/002.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/21-static_images-001.jpg" alt="Static / Stylized / Dining Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Dining Spaces</strong>
    <span>stylized/indoor/dining_spaces/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/22-static_images-000.jpg" alt="Static / Stylized / Living Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Living Spaces</strong>
    <span>stylized/indoor/living_spaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/23-static_images-000.jpg" alt="Static / Stylized / Passageways" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Passageways</strong>
    <span>stylized/indoor/passageways/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/24-static_images-000.jpg" alt="Static / Stylized / Public Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Public Spaces</strong>
    <span>stylized/indoor/public_spaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/25-static_images-000.jpg" alt="Static / Stylized / Workspaces" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Workspaces</strong>
    <span>stylized/indoor/workspaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/26-static_images-000.jpg" alt="Static / Stylized / Aquatic Landscape" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Aquatic Landscape</strong>
    <span>stylized/outdoor/aquatic_landscape/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/27-static_images-000.jpg" alt="Static / Stylized / City" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / City</strong>
    <span>stylized/outdoor/city/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/28-static_images-000.jpg" alt="Static / Stylized / Suburb" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Suburb</strong>
    <span>stylized/outdoor/suburb/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/29-static_images-000.jpg" alt="Static / Stylized / Terrestrial Landscape" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Terrestrial Landscape</strong>
    <span>stylized/outdoor/terrestrial_landscape/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/30-static_images-000.jpg" alt="Static / Stylized / Verdant Landscape" loading="lazy">
  <figcaption>
    <strong>Static / Stylized / Verdant Landscape</strong>
    <span>stylized/outdoor/verdant_landscape/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/31-dynamic_images-007.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/007.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/32-dynamic_images-016.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/016.png</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/33-dynamic_images-017.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/017.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/34-dynamic_images-021.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/021.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/35-dynamic_images-038.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/038.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/36-dynamic_images-043.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/043.jpg</span>
  </figcaption>
</figure>
</div>

## 图像分布

<iframe class="chart-frame" src="../../generated/charts/images-distribution.html" title="图像分布" loading="lazy"></iframe>

## 静态场景层级

<iframe class="chart-frame" src="../../generated/charts/images-static-scene-hierarchy.html" title="静态场景层级" loading="lazy"></iframe>

## Mask 覆盖

<iframe class="chart-frame" src="../../generated/charts/images-mask-coverage.html" title="Mask 覆盖" loading="lazy"></iframe>

## 分辨率分布

<iframe class="chart-frame" src="../../generated/charts/images-resolution-mix.html" title="分辨率分布" loading="lazy"></iframe>
