# 图像分析

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">图像行数</div>
  <div class="metric-value">9,518</div>
  <div class="metric-note">所有已扫描图像</div>
</div>
<div class="metric-card">
  <div class="metric-label">静态图像</div>
  <div class="metric-value">7,040</div>
  <div class="metric-note">以场景为中心的静帧</div>
</div>
<div class="metric-card">
  <div class="metric-label">动态图像</div>
  <div class="metric-value">2,478</div>
  <div class="metric-note">带运动标签的静帧</div>
</div>
<div class="metric-card">
  <div class="metric-label">带掩码的行</div>
  <div class="metric-value">248</div>
  <div class="metric-note">关联到分割掩码的图像</div>
</div>
</div>

## 静态场景汇总

| style          | environment   | scene                 |   items |
|:---------------|:--------------|:----------------------|--------:|
| photorealistic | indoor        | living_spaces         |    1051 |
| photorealistic | outdoor       | terrestrial_landscape |     694 |
| photorealistic | indoor        | dining_spaces         |     671 |
| photorealistic | outdoor       | city                  |     526 |
| photorealistic | outdoor       | verdant_landscape     |     416 |
| photorealistic | indoor        | public_spaces         |     401 |
| photorealistic | indoor        | workspaces            |     349 |
| stylized       | indoor        | living_spaces         |     271 |
| photorealistic | outdoor       | aquatic_landscape     |     271 |
| stylized       | outdoor       | terrestrial_landscape |     261 |
| stylized       | outdoor       | aquatic_landscape     |     241 |
| stylized       | indoor        | dining_spaces         |     241 |
| stylized       | indoor        | public_spaces         |     241 |
| stylized       | indoor        | workspaces            |     231 |
| stylized       | outdoor       | verdant_landscape     |     211 |

## 动态掩码汇总

| style          | motion_category   |   items |   masked |   mask_rate |
|:---------------|:------------------|--------:|---------:|------------:|
| photorealistic | articulated       |     420 |      112 |      0.2667 |
| photorealistic | deformable        |     257 |        0 |      0      |
| photorealistic | fluid             |     260 |       30 |      0.1154 |
| photorealistic | multi_motion      |     203 |        0 |      0      |
| photorealistic | rigid             |     109 |        0 |      0      |
| stylized       | articulated       |     255 |       66 |      0.2588 |
| stylized       | deformable        |     236 |       28 |      0.1186 |
| stylized       | fluid             |     251 |       12 |      0.0478 |
| stylized       | multi_motion      |     246 |        0 |      0      |
| stylized       | rigid             |     241 |        0 |      0      |

## 图像分布

<iframe class="chart-frame" src="../../generated/charts/images-distribution.html" title="Image Distribution" loading="lazy"></iframe>

## 静态场景层级

<iframe class="chart-frame" src="../../generated/charts/images-static-scene-hierarchy.html" title="Static Scene Hierarchy" loading="lazy"></iframe>

## 掩码覆盖率

<iframe class="chart-frame" src="../../generated/charts/images-mask-coverage.html" title="Mask Coverage" loading="lazy"></iframe>

## 分辨率构成

<iframe class="chart-frame" src="../../generated/charts/images-resolution-mix.html" title="Resolution Mix" loading="lazy"></iframe>
