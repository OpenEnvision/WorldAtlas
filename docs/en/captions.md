# Caption Analytics

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Captioned Media</div>
  <div class="metric-value">5,675</div>
  <div class="metric-note">Video and physics rows with captions</div>
</div>
<div class="metric-card">
  <div class="metric-label">Weather Tagged</div>
  <div class="metric-value">5,297</div>
  <div class="metric-note">Videos with normalized weather labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Time-Of-Day Tagged</div>
  <div class="metric-value">5,297</div>
  <div class="metric-note">Videos with normalized time-of-day labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Primary Scene Types</div>
  <div class="metric-value">128</div>
  <div class="metric-note">Unique normalized primary scene labels</div>
</div>
</div>

## Coverage Summary

| collection     |   items |   captions | caption_rate   | weather_rate   | time_of_day_rate   | brightness_rate   | crowd_density_rate   | scene_primary_rate   |
|:---------------|--------:|-----------:|:---------------|:---------------|:-------------------|:------------------|:---------------------|:---------------------|
| static_videos  |    4000 |       4000 | 100.00%        | 100.00%        | 100.00%            | 100.00%           | 100.00%              | 100.00%              |
| physics_videos |    1400 |        377 | 26.93%         | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |
| dynamic_videos |     798 |        798 | 100.00%        | 99.87%         | 99.87%             | 99.87%            | 99.87%               | 100.00%              |
| aaa_games      |     500 |        500 | 100.00%        | 100.00%        | 100.00%            | 100.00%           | 100.00%              | 100.00%              |

## Top Weather Labels

| value              |   items | share   |
|:-------------------|--------:|:--------|
| Clear              |    2670 | 50.41%  |
| Sunny              |     883 | 16.67%  |
| Cloudy             |     599 | 11.31%  |
| Unknown            |     536 | 10.12%  |
| Snowy              |     176 | 3.32%   |
| Rainy              |     165 | 3.11%   |
| Indoor/Unspecified |     110 | 2.08%   |
| Foggy              |      71 | 1.34%   |
| Overcast           |      36 | 0.68%   |
| Indeterminate      |      17 | 0.32%   |
| Stormy             |      10 | 0.19%   |
| Empty              |       8 | 0.15%   |

## Top Time-Of-Day Labels

| value                  |   items | share   |
|:-----------------------|--------:|:--------|
| Daytime                |    3867 | 73.00%  |
| Unknown                |     745 | 14.06%  |
| Dusk/Evening           |     277 | 5.23%   |
| Night                  |     214 | 4.04%   |
| Dawn                   |      89 | 1.68%   |
| Indeterminate          |      46 | 0.87%   |
| Sunset                 |      38 | 0.72%   |
| Morning                |       9 | 0.17%   |
| Transition (Dawn/Dusk) |       6 | 0.11%   |
| Indoor                 |       3 | 0.06%   |
| N/A                    |       1 | 0.02%   |
| Sunrise                |       1 | 0.02%   |

## Top Primary Scene Types

| value             |   items | share   |
|:------------------|--------:|:--------|
| Indoor            |    1089 | 20.55%  |
| Urban             |     687 | 12.97%  |
| Landscape         |     588 | 11.10%  |
| Interior          |     391 | 7.38%   |
| Nature            |     309 | 5.83%   |
| Natural Landscape |     288 | 5.44%   |
| Outdoor           |     225 | 4.25%   |
| Suburban          |     212 | 4.00%   |
| Natural           |     150 | 2.83%   |
| Forest            |     125 | 2.36%   |
| Rural             |     102 | 1.93%   |
| Village           |      78 | 1.47%   |
| Wildlife          |      77 | 1.45%   |
| Animal            |      76 | 1.43%   |
| Public Space      |      64 | 1.21%   |

## Caption Coverage

<iframe class="chart-frame" src="../generated/charts/captions-coverage.html" title="Caption Coverage" loading="lazy"></iframe>

## Weather Distribution

<iframe class="chart-frame" src="../generated/charts/captions-weather-by-collection.html" title="Weather Distribution" loading="lazy"></iframe>

## Time-Of-Day Distribution

<iframe class="chart-frame" src="../generated/charts/captions-time-of-day-by-collection.html" title="Time-Of-Day Distribution" loading="lazy"></iframe>

## Brightness And Crowd Density

<iframe class="chart-frame" src="../generated/charts/captions-attribute-mix.html" title="Brightness And Crowd Density" loading="lazy"></iframe>

## Primary Scene Types

<iframe class="chart-frame" src="../generated/charts/captions-scene-types.html" title="Primary Scene Types" loading="lazy"></iframe>

## Weather Versus Time Of Day

<iframe class="chart-frame" src="../generated/charts/captions-weather-time.html" title="Weather Versus Time Of Day" loading="lazy"></iframe>

## Brightness Versus Crowd Density

<iframe class="chart-frame" src="../generated/charts/captions-brightness-crowd.html" title="Brightness Versus Crowd Density" loading="lazy"></iframe>

## Artifacts

- [`caption_distribution_summary.csv`](../generated/data/caption_distribution_summary.csv)
- [`caption_collection_summary.csv`](../generated/data/caption_collection_summary.csv)
- [`caption_coverage_summary.csv`](../generated/data/caption_coverage_summary.csv)
- [`caption_weather_time_summary.csv`](../generated/data/caption_weather_time_summary.csv)
- [`caption_brightness_crowd_summary.csv`](../generated/data/caption_brightness_crowd_summary.csv)
