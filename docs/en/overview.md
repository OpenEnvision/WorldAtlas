# Dataset Overview

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Benchmark Items</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Primary benchmark rows</div>
</div>
<div class="metric-card">
  <div class="metric-label">Images</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Static and dynamic image assets</div>
</div>
<div class="metric-card">
  <div class="metric-label">Videos</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Static, dynamic, and gameplay videos</div>
</div>
<div class="metric-card">
  <div class="metric-label">Physics</div>
  <div class="metric-value">0</div>
  <div class="metric-note">Physics-focused video assets</div>
</div>
</div>

## Topology

```mermaid
graph TD
  root[WorldArena]
  root --> images[Image]
  root --> videos[Video]
  root --> physics[Physics]
  images --> static_images[Static Images • 0]
  images --> dynamic_images[Dynamic Images • 0]
  videos --> static_videos[Static Videos • 0]
  videos --> dynamic_videos[Dynamic Videos • 0]
  videos --> aaa_games[AAA Games • 0]
  physics --> physics_videos[Physics Videos • 0]
```

## Collection Summary

_No rows available._

## Collection Volume

<iframe class="chart-frame" src="../generated/charts/overview-collection-volume.html" title="Collection Volume" loading="lazy"></iframe>

## Benchmark Hierarchy

<iframe class="chart-frame" src="../generated/charts/overview-benchmark-hierarchy.html" title="Benchmark Hierarchy" loading="lazy"></iframe>







## Downloadable Artifacts

- [`summary.json`](../generated/data/summary.json)
- [`all_records.csv`](../generated/data/all_records.csv)
- [`image_records.csv`](../generated/data/image_records.csv)
- [`video_records.csv`](../generated/data/video_records.csv)
- [`physics_records.csv`](../generated/data/physics_records.csv)
