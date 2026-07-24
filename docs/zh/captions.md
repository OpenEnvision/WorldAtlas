# Caption 统计

!!! info "生成内容"
    当你新增、删除或重组源数据资产后，请运行 `worldarena build-docs --config config/datasets.yaml`。

## 概述

<p class="overview-lede">Caption 侧车为视频与 physics 条目附加 weather、time-of-day、brightness、crowd-density 与 scene-type 等标签。下列面板按集合汇总 caption 覆盖情况，再进入明细表与交互图表。</p>

## 集合规模

<p class="overview-section-lede">视频与 physics 集合的 headline caption 计数与归一化标签覆盖。</p>

<div class="overview-page">

<div class="overview-section">
  <p class="overview-section-label">总量指标</p>
  <p class="overview-section-note">主 caption 库存与归一化标签覆盖。</p>
<div class="overview-hero-grid">
<div class="overview-hero-card overview-hero-card--primary">
  <div class="overview-hero-label">带 Caption 的媒体</div>
  <div class="overview-hero-value">377</div>
  <div class="overview-hero-note">带 caption 的视频与 physics 条目</div>
</div>
<div class="overview-hero-card overview-hero-card--blue">
  <div class="overview-hero-label">Weather 标注</div>
  <div class="overview-hero-value">0</div>
  <div class="overview-hero-note">含归一化 weather 标签的视频</div>
</div>
<div class="overview-hero-card overview-hero-card--warm">
  <div class="overview-hero-label">Time-Of-Day 标注</div>
  <div class="overview-hero-value">0</div>
  <div class="overview-hero-note">含归一化 time-of-day 标签的视频</div>
</div>
<div class="overview-hero-card overview-hero-card--ink">
  <div class="overview-hero-label">主场景类型</div>
  <div class="overview-hero-value">0</div>
  <div class="overview-hero-note">不同归一化主场景标签</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">标签广度</p>
  <p class="overview-section-note">库存中不同的 weather、time-of-day、brightness 与 crowd-density 标签。</p>
<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Weather 标签</div>
  <div class="metric-value">0</div>
  <div class="metric-note">不同归一化 weather 标签</div>
</div>
<div class="metric-card">
  <div class="metric-label">Time-Of-Day 标签</div>
  <div class="metric-value">0</div>
  <div class="metric-note">不同归一化 time-of-day 标签</div>
</div>
<div class="metric-card">
  <div class="metric-label">Brightness 标签</div>
  <div class="metric-value">0</div>
  <div class="metric-note">不同归一化 brightness 标签</div>
</div>
<div class="metric-card">
  <div class="metric-label">Crowd Density 标签</div>
  <div class="metric-value">0</div>
  <div class="metric-note">不同归一化 crowd-density 标签</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">按集合的 caption 覆盖</p>
  <p class="overview-section-note">caption 侧车在 static、physics、dynamic 与 gameplay 集合中的分布。</p>
<div class="overview-regime-grid">
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">Physics 视频</span>
    <span class="overview-regime-share">26.9% 含 caption</span>
  </div>
  <div class="overview-regime-value">377</div>
  <div class="overview-regime-note">部分 caption 覆盖的仿真 physics 片段</div>
</div>
</div>
</div>

</div>

## 覆盖率汇总

| collection     |   items |   captions | caption_rate   | weather_rate   | time_of_day_rate   | brightness_rate   | crowd_density_rate   | scene_primary_rate   |
|:---------------|--------:|-----------:|:---------------|:---------------|:-------------------|:------------------|:---------------------|:---------------------|
| physics_videos |    1400 |        377 | 26.93%         | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |

## Caption 覆盖

<iframe class="chart-frame" src="../../generated/charts/captions-coverage.html" title="Caption 覆盖" loading="lazy"></iframe>
