# Caption 统计

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">带 Caption 的媒体</div>
  <div class="metric-value">873</div>
  <div class="metric-note">Video and physics rows with captions</div>
</div>
<div class="metric-card">
  <div class="metric-label">Weather Tagged</div>
  <div class="metric-value">496</div>
  <div class="metric-note">Videos with normalized weather labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Time-Of-Day Tagged</div>
  <div class="metric-value">496</div>
  <div class="metric-note">Videos with normalized time-of-day labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Primary Scene Types</div>
  <div class="metric-value">52</div>
  <div class="metric-note">Unique normalized primary scene labels</div>
</div>
</div>

## 覆盖率汇总

| collection     |   items |   captions | caption_rate   | weather_rate   | time_of_day_rate   | brightness_rate   | crowd_density_rate   | scene_primary_rate   |
|:---------------|--------:|-----------:|:---------------|:---------------|:-------------------|:------------------|:---------------------|:---------------------|
| static_videos  |    4000 |         31 | 0.78%          | 0.78%          | 0.78%              | 0.78%             | 0.78%                | 0.78%                |
| physics_videos |    1400 |        377 | 26.93%         | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |
| dynamic_videos |     788 |          0 | 0.00%          | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |
| aaa_games      |     465 |        465 | 100.00%        | 100.00%        | 100.00%            | 100.00%           | 100.00%              | 100.00%              |

## Top Weather Labels

| value   |   items | share   |
|:--------|--------:|:--------|
| Clear   |     294 | 59.27%  |
| Sunny   |      62 | 12.50%  |
| Cloudy  |      46 | 9.27%   |
| Foggy   |      29 | 5.85%   |
| Snowy   |      26 | 5.24%   |
| Rainy   |      20 | 4.03%   |
| Unknown |      13 | 2.62%   |
| Stormy  |       6 | 1.21%   |

## Top Time-Of-Day Labels

| value        |   items | share   |
|:-------------|--------:|:--------|
| Daytime      |     257 | 51.81%  |
| Dusk/Evening |      93 | 18.75%  |
| Unknown      |      65 | 13.10%  |
| Night        |      59 | 11.90%  |
| Dawn         |      22 | 4.44%   |

## Top Primary Scene Types

| value              |   items | share   |
|:-------------------|--------:|:--------|
| Forest             |     104 | 20.97%  |
| Village            |      59 | 11.90%  |
| Landscape          |      52 | 10.48%  |
| Urban              |      35 | 7.06%   |
| Indoor             |      31 | 6.25%   |
| Historical Fantasy |      24 | 4.84%   |
| Natural            |      18 | 3.63%   |
| Fantasy            |      17 | 3.43%   |
| Rural              |      15 | 3.02%   |
| Historical         |      14 | 2.82%   |
| Nature             |      14 | 2.82%   |
| Courtyard          |      10 | 2.02%   |
| Garden             |      10 | 2.02%   |
| Combat             |       7 | 1.41%   |
| Medieval           |       7 | 1.41%   |

## Caption Coverage

<iframe class="chart-frame" src="../../generated/charts/captions-coverage.html" title="Caption Coverage" loading="lazy"></iframe>

## Weather Distribution

<iframe class="chart-frame" src="../../generated/charts/captions-weather-by-collection.html" title="Weather Distribution" loading="lazy"></iframe>

## Time-Of-Day Distribution

<iframe class="chart-frame" src="../../generated/charts/captions-time-of-day-by-collection.html" title="Time-Of-Day Distribution" loading="lazy"></iframe>

## Brightness And Crowd Density

<iframe class="chart-frame" src="../../generated/charts/captions-attribute-mix.html" title="Brightness And Crowd Density" loading="lazy"></iframe>

## Primary Scene Types

<iframe class="chart-frame" src="../../generated/charts/captions-scene-types.html" title="Primary Scene Types" loading="lazy"></iframe>

## Weather Versus Time Of Day

<iframe class="chart-frame" src="../../generated/charts/captions-weather-time.html" title="Weather Versus Time Of Day" loading="lazy"></iframe>

## Brightness Versus Crowd Density

<iframe class="chart-frame" src="../../generated/charts/captions-brightness-crowd.html" title="Brightness Versus Crowd Density" loading="lazy"></iframe>

## 产物

- [`caption_distribution_summary.csv`](../generated/data/caption_distribution_summary.csv)
- [`caption_collection_summary.csv`](../generated/data/caption_collection_summary.csv)
- [`caption_coverage_summary.csv`](../generated/data/caption_coverage_summary.csv)
- [`caption_weather_time_summary.csv`](../generated/data/caption_weather_time_summary.csv)
- [`caption_brightness_crowd_summary.csv`](../generated/data/caption_brightness_crowd_summary.csv)
