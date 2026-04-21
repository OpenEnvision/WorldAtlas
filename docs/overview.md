# Dataset Overview

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Benchmark Items</div>
  <div class="metric-value">19,980</div>
  <div class="metric-note">Primary benchmark rows</div>
</div>
<div class="metric-card">
  <div class="metric-label">Images</div>
  <div class="metric-value">9,518</div>
  <div class="metric-note">Static and dynamic image assets</div>
</div>
<div class="metric-card">
  <div class="metric-label">Videos</div>
  <div class="metric-value">10,550</div>
  <div class="metric-note">Benchmark and auxiliary video assets</div>
</div>
<div class="metric-card">
  <div class="metric-label">Annotated Videos</div>
  <div class="metric-value">6,250</div>
  <div class="metric-note">Rows with SpatialVID-style metadata</div>
</div>
</div>

## Topology

```mermaid
graph TD
  root[WorldArena]
  root --> benchmark[Benchmark]
  root --> auxiliary[Auxiliary]
  benchmark --> images[Images]
  benchmark --> videos[Videos]
  images --> static_images[Static Images • 7040]
  images --> dynamic_images[Dynamic Images • 2478]
  videos --> static_videos[Static Videos • 8991]
  videos --> dynamic_videos[Dynamic Videos • 971]
  videos --> aaa_games[AAA Games • 500]
  auxiliary --> review_assets[Review Assets • 88]
```

## Collection Summary

| group     | collection     | modality   |   items |   annotated |   masked |
|:----------|:---------------|:-----------|--------:|------------:|---------:|
| auxiliary | review_assets  | video      |      88 |           0 |        0 |
| benchmark | dynamic_images | image      |    2478 |           0 |      248 |
| benchmark | static_images  | image      |    7040 |           0 |        0 |
| benchmark | aaa_games      | video      |     500 |         500 |      500 |
| benchmark | dynamic_videos | video      |     971 |         319 |      319 |
| benchmark | static_videos  | video      |    8991 |        5431 |     5431 |

## Collection Volume

<iframe class="chart-frame" src="../generated/charts/overview-collection-volume.html" title="Collection Volume" loading="lazy"></iframe>

## Benchmark Hierarchy

<iframe class="chart-frame" src="../generated/charts/overview-benchmark-hierarchy.html" title="Benchmark Hierarchy" loading="lazy"></iframe>

## Downloadable Artifacts

- [`summary.json`](generated/data/summary.json)
- [`all_records.csv`](generated/data/all_records.csv)
- [`image_records.csv`](generated/data/image_records.csv)
- [`video_records.csv`](generated/data/video_records.csv)
