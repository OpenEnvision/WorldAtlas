# Dataset Overview

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

<p class="doc-provenance">Snapshot note: panels and charts below use cached `docs/generated/` artifacts in this checkout because live dataset roots were unavailable during the last `worldarena build-docs` run. Re-run against local dataset roots when a live rescan is available.</p>

## Summary

<p class="overview-lede">WorldAtlas Arena combines a multi-regime benchmark corpus, a six-axis evaluation contract, and a registry of world-model adapters. The panels below summarize scale from complementary angles—media inventory, task regime, metric surface, and model coverage—before the collection tables and sample galleries.</p>

## Benchmark Scale

<p class="overview-section-lede">Headline totals, media inventory, task regimes, and benchmark contract surface.</p>

<div class="overview-page">

<div class="overview-section">
  <p class="overview-section-label">Headline totals</p>
  <p class="overview-section-note">Primary inventory counts, suites, metrics, and model coverage.</p>
<div class="overview-hero-grid">
<div class="overview-hero-card overview-hero-card--primary">
  <div class="overview-hero-label">Benchmark Items</div>
  <div class="overview-hero-value">11,653</div>
  <div class="overview-hero-note">Primary rows across image, video, and physics suites</div>
</div>
<div class="overview-hero-card overview-hero-card--green">
  <div class="overview-hero-label">Formal Suites</div>
  <div class="overview-hero-value">4</div>
  <div class="overview-hero-note">Public benchmark contracts in `config/benchmark.yaml`</div>
</div>
<div class="overview-hero-card overview-hero-card--warm">
  <div class="overview-hero-label">Documented Metrics</div>
  <div class="overview-hero-value">47</div>
  <div class="overview-hero-note">Public metric inventory in the benchmark docs</div>
</div>
<div class="overview-hero-card overview-hero-card--blue">
  <div class="overview-hero-label">Evaluation Dimensions</div>
  <div class="overview-hero-value">6</div>
  <div class="overview-hero-note">Axes aggregated in `per_dimension_summary.json`</div>
</div>
<div class="overview-hero-card overview-hero-card--ink">
  <div class="overview-hero-label">Registered Models</div>
  <div class="overview-hero-value">25</div>
  <div class="overview-hero-note">Adapter configs under `config/models/`</div>
</div>
<div class="overview-hero-card overview-hero-card--green">
  <div class="overview-hero-label">Annotated Rows</div>
  <div class="overview-hero-value">5,496</div>
  <div class="overview-hero-note">47.2% of benchmark rows carry supervision sidecars</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">Media inventory</p>
  <p class="overview-section-note">Image, video, and physics assets in the benchmark corpus.</p>
<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Images</div>
  <div class="metric-value">5,000</div>
  <div class="metric-note">Static and dynamic image assets</div>
</div>
<div class="metric-card">
  <div class="metric-label">Videos</div>
  <div class="metric-value">5,253</div>
  <div class="metric-note">Static, dynamic, and gameplay videos</div>
</div>
<div class="metric-card">
  <div class="metric-label">Physics</div>
  <div class="metric-value">1,400</div>
  <div class="metric-note">Physics-focused video assets</div>
</div>
<div class="metric-card">
  <div class="metric-label">Masked Dynamics</div>
  <div class="metric-value">2,124</div>
  <div class="metric-note">Dynamic image and gameplay rows with segmentation masks</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">Task regimes</p>
  <p class="overview-section-note">How benchmark rows split across static, dynamic, interaction, and physics tracks.</p>
<div class="overview-regime-grid">
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">Static world generation</span>
    <span class="overview-regime-share">63.3%</span>
  </div>
  <div class="overview-regime-value">7,372</div>
  <div class="overview-regime-note">Camera or layout change with scene preservation</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">Dynamic world generation</span>
    <span class="overview-regime-share">20.7%</span>
  </div>
  <div class="overview-regime-value">2,416</div>
  <div class="overview-regime-note">In-scene temporal evolution under controlled viewpoints</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">Interaction-conditioned</span>
    <span class="overview-regime-share">4.0%</span>
  </div>
  <div class="overview-regime-value">465</div>
  <div class="overview-regime-note">Gameplay and action-conditioned rollouts</div>
</div>
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">Physics metrics</span>
    <span class="overview-regime-share">12.0%</span>
  </div>
  <div class="overview-regime-value">1,400</div>
  <div class="overview-regime-note">Reference motion and simulator-backed physics tracks</div>
</div>
</div>
</div>

<div class="overview-section">
  <p class="overview-section-label">Evaluation Framework</p>
  <p class="overview-section-note">Formal suites separate static, dynamic, and interaction-conditioned generation while sharing a common reporting contract. Each suite declares an applicable metric vector in `config/benchmark.yaml`.</p>

<table>
<thead>
<tr><th align="left">suite</th><th align="left">modality</th><th align="left">regime</th><th align="right">metrics</th></tr>
</thead>
<tbody>
<tr><td align="left">image_static</td><td align="left">image → video</td><td align="left">static</td><td align="right">43</td></tr>
<tr><td align="left">image_dynamic</td><td align="left">image → video</td><td align="left">dynamic</td><td align="right">33</td></tr>
<tr><td align="left">video_static</td><td align="left">video → video</td><td align="left">static</td><td align="right">37</td></tr>
<tr><td align="left">video_dynamic</td><td align="left">video → video</td><td align="left">dynamic</td><td align="right">43</td></tr>
</tbody>
</table>

  <p class="overview-section-label">Metrics by Dimension</p>

<div class="overview-dimension-bars">
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">Long Sequence</span>
    <span class="overview-dimension-count">1 metrics · 0 official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--long_sequence" style="width: 8%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">Action Control</span>
    <span class="overview-dimension-count">12 metrics · 4 official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--action_control" style="width: 86%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">3D/4D Consistency</span>
    <span class="overview-dimension-count">7 metrics · 2 official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--consistency_3d_4d" style="width: 50%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">Physics</span>
    <span class="overview-dimension-count">11 metrics · 10 official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--physics" style="width: 79%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">Quality</span>
    <span class="overview-dimension-count">14 metrics · 5 official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--quality" style="width: 100%;"></div>
  </div>
</div>
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">Real Time</span>
    <span class="overview-dimension-count">2 metrics · 2 official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--real_time" style="width: 14%;"></div>
  </div>
</div>
</div>

  <p class="overview-section-label">Official vs Diagnostic Surface</p>

<div class="overview-split-grid">
<div class="overview-split-card overview-split-card--official">
  <div class="overview-split-label">Official surface</div>
  <div class="overview-split-value">23</div>
  <div class="overview-split-note">Headline or conditionally official metrics in the public inventory</div>
</div>
<div class="overview-split-card overview-split-card--supplementary">
  <div class="overview-split-label">Diagnostic probes</div>
  <div class="overview-split-value">24</div>
  <div class="overview-split-note">Supplemental geometry, reconstruction, and long-horizon probes</div>
</div>
<div class="overview-split-card overview-split-card--experimental">
  <div class="overview-split-label">Experimental suite</div>
  <div class="overview-split-value">19</div>
  <div class="overview-split-note">Non-headline slice declared separately in `config/benchmark.yaml`</div>
</div>
</div>

  <p class="overview-section-note">See the <a href="benchmark/metrics/index.md">Metrics Overview</a> for eligibility rules, normalization, and aggregation semantics.</p>

  <div class="overview-quick-links">
    <a href="benchmark/metrics/index.md">Metrics Overview</a>
    <a href="benchmark/index.md">Benchmark Overview</a>
    <a href="development.md">Development</a>
  </div>
</div>

</div>

## Topology

```mermaid
graph TD
  root[WorldAtlas Arena]
  root --> images[Image]
  root --> videos[Video]
  root --> physics[Physics]
  images --> static_images[Static Images • 3372]
  images --> dynamic_images[Dynamic Images • 1628]
  videos --> static_videos[Static Videos • 4000]
  videos --> dynamic_videos[Dynamic Videos • 788]
  videos --> aaa_games[AAA Games • 465]
  physics --> physics_videos[Physics Videos • 1400]
```

## Collection Summary

| group     | collection     | modality   |   items |   annotated |   masked |
|:----------|:---------------|:-----------|--------:|------------:|---------:|
| benchmark | dynamic_images | image      |    1628 |        1628 |     1628 |
| benchmark | static_images  | image      |    3372 |        3372 |        0 |
| benchmark | physics_videos | physics    |    1400 |           0 |        0 |
| benchmark | aaa_games      | video      |     465 |         465 |      465 |
| benchmark | dynamic_videos | video      |     788 |           0 |        0 |
| benchmark | static_videos  | video      |    4000 |          31 |       31 |

## Collection Volume

<iframe class="chart-frame" src="../generated/charts/overview-collection-volume.html" title="Collection Volume" loading="lazy"></iframe>

## Benchmark Hierarchy

<iframe class="chart-frame" src="../generated/charts/overview-benchmark-hierarchy.html" title="Benchmark Hierarchy" loading="lazy"></iframe>

## Sample Image Gallery

<div class="sample-grid sample-grid--image">
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/01-dynamic_images-003.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/003.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/02-static_images-000.jpg" alt="Static / Photorealistic / Dining Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Dining Spaces</strong>
    <span>photorealistic/indoor/dining_spaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/03-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Deformable</strong>
    <span>photorealistic/images/deformable/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/04-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Fluid</strong>
    <span>photorealistic/images/fluid/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/05-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Multi Motion</strong>
    <span>photorealistic/images/multi_motion/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/06-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Rigid</strong>
    <span>photorealistic/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/07-dynamic_images-000.jpg" alt="Dynamic / Stylized / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Articulated</strong>
    <span>stylized/images/articulated/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/08-dynamic_images-001.jpg" alt="Dynamic / Stylized / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Deformable</strong>
    <span>stylized/images/deformable/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/09-dynamic_images-000.jpg" alt="Dynamic / Stylized / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Fluid</strong>
    <span>stylized/images/fluid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/10-dynamic_images-000.jpg" alt="Dynamic / Stylized / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Multi Motion</strong>
    <span>stylized/images/multi_motion/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/11-dynamic_images-000.jpg" alt="Dynamic / Stylized / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Rigid</strong>
    <span>stylized/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/images/12-static_images-0000.jpg" alt="Static / Photorealistic / Living Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Living Spaces</strong>
    <span>photorealistic/indoor/living_spaces/0000.png</span>
  </figcaption>
</figure>
</div>



## Sample Physics Gallery

<div class="sample-grid sample-grid--physics">
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/posters/01-physics_videos-00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.jpg" alt="Basic Physics Consistency / Acceleration" loading="lazy">
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../generated/samples/posters/15-physics_videos-06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.jpg" alt="Basic Physics Consistency / Uniform Motion" loading="lazy">
  <figcaption>
    <strong>Basic Physics Consistency / Uniform Motion</strong>
    <span>videos_by_category/basic_physics_consistency/uniform_motion/06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.mp4</span>
  </figcaption>
</figure>
</div>
