# Caption Analytics

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Captioned Videos</div>
  <div class="metric-value">6,250</div>
  <div class="metric-note">Videos with caption metadata</div>
</div>
<div class="metric-card">
  <div class="metric-label">Weather Tagged</div>
  <div class="metric-value">6,248</div>
  <div class="metric-note">Videos with normalized weather labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Time-Of-Day Tagged</div>
  <div class="metric-value">6,248</div>
  <div class="metric-note">Videos with normalized time-of-day labels</div>
</div>
<div class="metric-card">
  <div class="metric-label">Primary Scene Types</div>
  <div class="metric-value">89</div>
  <div class="metric-note">Unique normalized primary scene labels</div>
</div>
</div>

## Coverage Summary

| collection     |   items |   captions | caption_rate   | weather_rate   | time_of_day_rate   | brightness_rate   | crowd_density_rate   | scene_primary_rate   |
|:---------------|--------:|-----------:|:---------------|:---------------|:-------------------|:------------------|:---------------------|:---------------------|
| static_videos  |    8991 |       5431 | 60.40%         | 60.39%         | 60.39%             | 60.39%            | 60.39%               | 60.40%               |
| dynamic_videos |     971 |        319 | 32.85%         | 32.75%         | 32.75%             | 32.75%            | 32.75%               | 32.85%               |
| aaa_games      |     500 |        500 | 100.00%        | 100.00%        | 100.00%            | 100.00%           | 100.00%              | 100.00%              |
| review_assets  |      88 |          0 | 0.00%          | 0.00%          | 0.00%              | 0.00%             | 0.00%                | 0.00%                |

## Top Weather Labels

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

## Top Time-Of-Day Labels

| value                  |   items | share   |
|:-----------------------|--------:|:--------|
| Daytime                |    4400 | 70.42%  |
| Unknown                |    1070 | 17.13%  |
| Dusk/Evening           |     358 | 5.73%   |
| Night                  |     282 | 4.51%   |
| Dawn                   |     126 | 2.02%   |
| Morning                |       7 | 0.11%   |
| Transition (Dawn/Dusk) |       5 | 0.08%   |

## Top Primary Scene Types

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
