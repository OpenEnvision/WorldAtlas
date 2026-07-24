# Caption Analytics

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

## Summary

<p class="overview-lede">Caption sidecars attach weather, time-of-day, brightness, crowd-density, and scene-type labels to video and physics rows. Panels below summarize caption coverage by collection before the detailed tables and interactive charts.</p>

## Collection Scale

<p class="overview-section-lede">Headline caption counts and normalized label coverage across video and physics collections.</p>

<div class="overview-page">

<div class="overview-section">
  <p class="overview-section-label">Headline totals</p>
  <p class="overview-section-note">Primary caption inventory and normalized label coverage.</p>
<div class="overview-hero-grid">
<div class="overview-hero-card overview-hero-card--primary">
  <div class="overview-hero-label">Captioned Media</div>
  <div class="overview-hero-value">377</div>
  <div class="overview-hero-note">Video and physics rows with captions</div>
</div>
<div class="overview-hero-card overview-hero-card--blue">
  <div class="overview-hero-label">Weather Tagged</div>
  <div class="overview-hero-value">0</div>
  <div class="overview-hero-note">Videos with normalized weather labels</div>
</div>
<div class="overview-hero-card overview-hero-card--warm">
  <div class="overview-hero-label">Time-Of-Day Tagged</div>
  <div class="overview-hero-value">0</div>
  <div class="overview-hero-note">Videos with normalized time-of-day labels</div>
</div>
<div class="overview-hero-card overview-hero-card--ink">
  <div class="overview-hero-label">Primary Scene Types</div>
  <div class="overview-hero-value">0</div>
  <div class="overview-hero-note">Unique normalized primary scene labels</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">Label breadth</p>
  <p class="overview-section-note">Distinct weather, time-of-day, brightness, and crowd-density labels in inventory.</p>
<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Weather Labels</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Distinct normalized weather labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Time-Of-Day Labels</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Distinct normalized time-of-day labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Brightness Labels</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Distinct normalized brightness labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Crowd Density Labels</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Distinct normalized crowd-density labels</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">Caption coverage by collection</p>
  <p class="overview-section-note">How caption sidecars distribute across static, physics, dynamic, and gameplay collections.</p>
<div class="overview-regime-grid">
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">Physics Videos</span>
    <span class="overview-regime-share">26.9% captioned</span>
  </div>
  <div class="overview-regime-value">377</div>
  <div class="overview-regime-note">Simulator-backed physics clips with partial caption coverage</div>
</div>
</div>
</div>

</div>

## Coverage Summary

| collection     |   items |   captions | caption_rate   | weather_rate   | time_of_day_rate   | brightness_rate   | crowd_density_rate   | scene_primary_rate   |
|:---------------|--------:|-----------:|:---------------|:---------------|:-------------------|:------------------|:---------------------|:---------------------|
| physics_videos |    1400 |        377 | 26.93%         | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |

## Caption Coverage

<iframe class="chart-frame" src="../generated/charts/captions-coverage.html" title="Caption Coverage" loading="lazy"></iframe>
