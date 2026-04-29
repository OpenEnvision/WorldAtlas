# Caption 分析

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">带 Caption 的视频</div>
  <div class="metric-value">6,250</div>
  <div class="metric-note">拥有 caption 元数据的视频</div>
</div>
<div class="metric-card">
  <div class="metric-label">带天气标签</div>
  <div class="metric-value">6,248</div>
  <div class="metric-note">具有归一化天气标签的视频</div>
</div>
<div class="metric-card">
  <div class="metric-label">带昼夜标签</div>
  <div class="metric-value">6,248</div>
  <div class="metric-note">具有归一化时间段标签的视频</div>
</div>
<div class="metric-card">
  <div class="metric-label">主要场景类型</div>
  <div class="metric-value">89</div>
  <div class="metric-note">唯一的归一化主场景标签数量</div>
</div>
</div>

## 覆盖率汇总

| collection     |   items |   captions | caption_rate   | weather_rate   | time_of_day_rate   | brightness_rate   | crowd_density_rate   | scene_primary_rate   |
|:---------------|--------:|-----------:|:---------------|:---------------|:-------------------|:------------------|:---------------------|:---------------------|
| static_videos  |    8991 |       5431 | 60.40%         | 60.39%         | 60.39%             | 60.39%            | 60.39%               | 60.40%               |
| dynamic_videos |     971 |        319 | 32.85%         | 32.75%         | 32.75%             | 32.75%            | 32.75%               | 32.85%               |
| aaa_games      |     500 |        500 | 100.00%        | 100.00%        | 100.00%            | 100.00%           | 100.00%              | 100.00%              |
| review_assets  |      88 |          0 | 0.00%          | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |

## 高频天气标签

| value              |   items | share   |
|:-------------------|--------:|:--------|
| Clear              |    2607 | 41.73%  |
| Sunny              |    1293 | 20.69%  |
| Unknown            |     991 | 15.86%  |
| Cloudy             |     821 | 13.14%  |
| Rainy              |     261 | 4.18%   |
| Snowy              |     188 | 3.01%   |
| Foggy              |      77 | 1.23%   |
| Stormy             |       9 | 0.14%   |
| Indoor/Unspecified |       1 | 0.02%   |

## 高频时间段标签

| value                  |   items | share   |
|:-----------------------|--------:|:--------|
| Daytime                |    4400 | 70.42%  |
| Unknown                |    1070 | 17.13%  |
| Dusk/Evening           |     358 | 5.73%   |
| Night                  |     282 | 4.51%   |
| Dawn                   |     126 | 2.02%   |
| Morning                |       7 | 0.11%   |
| Transition (Dawn/Dusk) |       5 | 0.08%   |

## 高频主要场景类型

| value             |   items | share   |
|:------------------|--------:|:--------|
| Indoor            |    1785 | 28.56%  |
| Urban             |     764 | 12.22%  |
| Landscape         |     557 | 8.91%   |
| Interior          |     536 | 8.58%   |
| Natural Landscape |     414 | 6.62%   |
| Outdoor           |     344 | 5.50%   |
| Suburban          |     340 | 5.44%   |
| Natural           |     207 | 3.31%   |
| Public Space      |     153 | 2.45%   |
| Rural             |     130 | 2.08%   |
| Forest            |     118 | 1.89%   |
| Nature            |     109 | 1.74%   |
| Animal            |      83 | 1.33%   |
| Aquatic           |      74 | 1.18%   |
| Waterfront        |      73 | 1.17%   |

## Caption 覆盖率

<iframe class="chart-frame" src="../../generated/charts/captions-coverage.html" title="Caption Coverage" loading="lazy"></iframe>

## 天气分布

<iframe class="chart-frame" src="../../generated/charts/captions-weather-by-collection.html" title="Weather Distribution" loading="lazy"></iframe>

## 时间段分布

<iframe class="chart-frame" src="../../generated/charts/captions-time-of-day-by-collection.html" title="Time-Of-Day Distribution" loading="lazy"></iframe>

## 亮度与人群密度

<iframe class="chart-frame" src="../../generated/charts/captions-attribute-mix.html" title="Brightness And Crowd Density" loading="lazy"></iframe>

## 主要场景类型

<iframe class="chart-frame" src="../../generated/charts/captions-scene-types.html" title="Primary Scene Types" loading="lazy"></iframe>

## 天气与时间段交叉

<iframe class="chart-frame" src="../../generated/charts/captions-weather-time.html" title="Weather Versus Time Of Day" loading="lazy"></iframe>

## 亮度与人群密度交叉

<iframe class="chart-frame" src="../../generated/charts/captions-brightness-crowd.html" title="Brightness Versus Crowd Density" loading="lazy"></iframe>

## 产物

- [`caption_distribution_summary.csv`](../generated/data/caption_distribution_summary.csv)
- [`caption_collection_summary.csv`](../generated/data/caption_collection_summary.csv)
- [`caption_coverage_summary.csv`](../generated/data/caption_coverage_summary.csv)
- [`caption_weather_time_summary.csv`](../generated/data/caption_weather_time_summary.csv)
- [`caption_brightness_crowd_summary.csv`](../generated/data/caption_brightness_crowd_summary.csv)
