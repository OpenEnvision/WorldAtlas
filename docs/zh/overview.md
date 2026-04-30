# 数据概览

!!! info "生成内容"
    当你新增、删除或重组源数据资产后，请运行 `worldarena build-docs --config config/datasets.yaml`。

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">基准条目数</div>
  <div class="metric-value">11,698</div>
  <div class="metric-note">Primary benchmark rows</div>
</div>
<div class="metric-card">
  <div class="metric-label">Images</div>
  <div class="metric-value">5,000</div>
  <div class="metric-note">Static and dynamic image assets</div>
</div>
<div class="metric-card">
  <div class="metric-label">Videos</div>
  <div class="metric-value">5,298</div>
  <div class="metric-note">Static, dynamic, and gameplay videos</div>
</div>
<div class="metric-card">
  <div class="metric-label">Physics</div>
  <div class="metric-value">1,400</div>
  <div class="metric-note">Physics-focused video assets</div>
</div>
</div>

## 拓扑

```mermaid
graph TD
  root[WorldArena]
  root --> images[Image]
  root --> videos[Video]
  root --> physics[Physics]
  images --> static_images[静态图像 • 3372]
  images --> dynamic_images[动态图像 • 1628]
  videos --> static_videos[Static Videos • 4000]
  videos --> dynamic_videos[Dynamic Videos • 798]
  videos --> aaa_games[AAA Games • 500]
  physics --> physics_videos[Physics Videos • 1400]
```

## 集合汇总

| group     | collection     | modality   |   items |   annotated |   masked |
|:----------|:---------------|:-----------|--------:|------------:|---------:|
| benchmark | dynamic_images | image      |    1628 |        1628 |     1628 |
| benchmark | static_images  | image      |    3372 |        3372 |        0 |
| benchmark | physics_videos | physics    |    1400 |           0 |        0 |
| benchmark | aaa_games      | video      |     500 |         500 |      500 |
| benchmark | dynamic_videos | video      |     798 |         798 |      307 |
| benchmark | static_videos  | video      |    4000 |        4000 |     2949 |

## Collection Volume

<iframe class="chart-frame" src="../../generated/charts/overview-collection-volume.html" title="Collection Volume" loading="lazy"></iframe>

## Benchmark Hierarchy

<iframe class="chart-frame" src="../../generated/charts/overview-benchmark-hierarchy.html" title="Benchmark Hierarchy" loading="lazy"></iframe>

## 图像样例

<div class="sample-grid sample-grid--image">
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/01-dynamic_images-003.jpg" alt="Dynamic / Photorealistic / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Articulated</strong>
    <span>photorealistic/images/articulated/003.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/02-static_images-000.jpg" alt="Static / Photorealistic / Dining Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Dining Spaces</strong>
    <span>photorealistic/indoor/dining_spaces/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/03-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Deformable</strong>
    <span>photorealistic/images/deformable/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/04-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Fluid</strong>
    <span>photorealistic/images/fluid/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/05-dynamic_images-001.jpg" alt="Dynamic / Photorealistic / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Multi Motion</strong>
    <span>photorealistic/images/multi_motion/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/06-dynamic_images-000.jpg" alt="Dynamic / Photorealistic / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Photorealistic / Rigid</strong>
    <span>photorealistic/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/07-dynamic_images-000.jpg" alt="Dynamic / Stylized / Articulated" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Articulated</strong>
    <span>stylized/images/articulated/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/08-dynamic_images-001.jpg" alt="Dynamic / Stylized / Deformable" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Deformable</strong>
    <span>stylized/images/deformable/001.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/09-dynamic_images-000.jpg" alt="Dynamic / Stylized / Fluid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Fluid</strong>
    <span>stylized/images/fluid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/10-dynamic_images-000.jpg" alt="Dynamic / Stylized / Multi Motion" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Multi Motion</strong>
    <span>stylized/images/multi_motion/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/11-dynamic_images-000.jpg" alt="Dynamic / Stylized / Rigid" loading="lazy">
  <figcaption>
    <strong>Dynamic / Stylized / Rigid</strong>
    <span>stylized/images/rigid/000.jpg</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <img class="sample-media" src="../../generated/samples/images/12-static_images-0000.jpg" alt="Static / Photorealistic / Living Spaces" loading="lazy">
  <figcaption>
    <strong>Static / Photorealistic / Living Spaces</strong>
    <span>photorealistic/indoor/living_spaces/0000.png</span>
  </figcaption>
</figure>
</div>

## 视频样例

<div class="sample-grid sample-grid--video">
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/01-aaa_games-000.jpg"><source src="../../generated/samples/previews/01-aaa_games-000.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/000.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/02-dynamic_videos-004787088a21__pexels_7486504-hd_1920_1080_30fps.jpg"><source src="../../generated/samples/previews/02-dynamic_videos-004787088a21__pexels_7486504-hd_1920_1080_30fps.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Dynamic / Articulated</strong>
    <span>photorealistic/videos/articulated/004787088a21__pexels_7486504-hd_1920_1080_30fps.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/03-static_videos-002b00c5-12a7-5b15-bbae-f05ad3544d7e.jpg"><source src="../../generated/samples/previews/03-static_videos-002b00c5-12a7-5b15-bbae-f05ad3544d7e.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Dining Spaces</strong>
    <span>photorealistic/indoor/dining_spaces/0-5s/002b00c5-12a7-5b15-bbae-f05ad3544d7e.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/04-dynamic_videos-0003028c30e6__2414312-sd_640_360_30fps.jpg"><source src="../../generated/samples/previews/04-dynamic_videos-0003028c30e6__2414312-sd_640_360_30fps.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Dynamic / Deformable</strong>
    <span>photorealistic/videos/deformable/0003028c30e6__2414312-sd_640_360_30fps.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/05-dynamic_videos-00f1d568f881__pexels_9722325-hd_2048_872_25fps.jpg"><source src="../../generated/samples/previews/05-dynamic_videos-00f1d568f881__pexels_9722325-hd_2048_872_25fps.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Dynamic / Fluid</strong>
    <span>photorealistic/videos/fluid/00f1d568f881__pexels_9722325-hd_2048_872_25fps.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/06-dynamic_videos-0427f9886ee5__pexels_35844496-hd_3840_2160_30fps.jpg"><source src="../../generated/samples/previews/06-dynamic_videos-0427f9886ee5__pexels_35844496-hd_3840_2160_30fps.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Dynamic / Multi Motion</strong>
    <span>photorealistic/videos/multi_motion/0427f9886ee5__pexels_35844496-hd_3840_2160_30fps.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/07-dynamic_videos-011845f3-2474-4b15-9151-157a5a1d95b5.jpg"><source src="../../generated/samples/previews/07-dynamic_videos-011845f3-2474-4b15-9151-157a5a1d95b5.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Dynamic / Rigid</strong>
    <span>photorealistic/videos/rigid/011845f3-2474-4b15-9151-157a5a1d95b5.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/08-static_videos-0092084d-dd8d-57fd-9b9e-fdf66b21e09e.jpg"><source src="../../generated/samples/previews/08-static_videos-0092084d-dd8d-57fd-9b9e-fdf66b21e09e.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Living Spaces</strong>
    <span>photorealistic/indoor/living_spaces/0-5s/0092084d-dd8d-57fd-9b9e-fdf66b21e09e.mp4</span>
  </figcaption>
</figure>
</div>

## Physics 样例

<div class="sample-grid sample-grid--physics">
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/01-physics_videos-00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.jpg"><source src="../../generated/samples/previews/01-physics_videos-00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/02-physics_videos-099a31d25b7b9f27d8dca0d86c286737761571d8c20aa8f45d81c1ce1013c162.jpg"><source src="../../generated/samples/previews/02-physics_videos-099a31d25b7b9f27d8dca0d86c286737761571d8c20aa8f45d81c1ce1013c162.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Circular Motion</strong>
    <span>videos_by_category/basic_physics_consistency/circular_motion/099a31d25b7b9f27d8dca0d86c286737761571d8c20aa8f45d81c1ce1013c162.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/03-physics_videos-00e24098bbcee584cb7e8d5d200186b5dfe38b978ec8cfff842f417c520cc967.jpg"><source src="../../generated/samples/previews/03-physics_videos-00e24098bbcee584cb7e8d5d200186b5dfe38b978ec8cfff842f417c520cc967.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Collision</strong>
    <span>videos_by_category/basic_physics_consistency/collision/00e24098bbcee584cb7e8d5d200186b5dfe38b978ec8cfff842f417c520cc967.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/04-physics_videos-01891e77e3e9afc366ec41a71d063409c988c3a4a572232e0a3cd3425de31281.jpg"><source src="../../generated/samples/previews/04-physics_videos-01891e77e3e9afc366ec41a71d063409c988c3a4a572232e0a3cd3425de31281.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Damped Oscillation</strong>
    <span>videos_by_category/basic_physics_consistency/damped_oscillation/01891e77e3e9afc366ec41a71d063409c988c3a4a572232e0a3cd3425de31281.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/05-physics_videos-019f9c756bb4c7260ad87e2d38469dc5b99287bfdf24438c1975a8aad97771ab.jpg"><source src="../../generated/samples/previews/05-physics_videos-019f9c756bb4c7260ad87e2d38469dc5b99287bfdf24438c1975a8aad97771ab.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Deceleration</strong>
    <span>videos_by_category/basic_physics_consistency/deceleration/019f9c756bb4c7260ad87e2d38469dc5b99287bfdf24438c1975a8aad97771ab.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/06-physics_videos-0185106d958eafa893e759d16434ed1ce35482526e5edd7d94d6f01fe2b4c988.jpg"><source src="../../generated/samples/previews/06-physics_videos-0185106d958eafa893e759d16434ed1ce35482526e5edd7d94d6f01fe2b4c988.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Deformation</strong>
    <span>videos_by_category/basic_physics_consistency/deformation/0185106d958eafa893e759d16434ed1ce35482526e5edd7d94d6f01fe2b4c988.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/07-physics_videos-0a30d1aa96460386a859a803ec80d60901dd453be737844250cb9c134d7119e2.jpg"><source src="../../generated/samples/previews/07-physics_videos-0a30d1aa96460386a859a803ec80d60901dd453be737844250cb9c134d7119e2.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Elastic Motion</strong>
    <span>videos_by_category/basic_physics_consistency/elastic_motion/0a30d1aa96460386a859a803ec80d60901dd453be737844250cb9c134d7119e2.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/08-physics_videos-082e01d1b61f9360305144d49feea870487197c2560475116c79b8425bdde234.jpg"><source src="../../generated/samples/previews/08-physics_videos-082e01d1b61f9360305144d49feea870487197c2560475116c79b8425bdde234.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Motion 3D</strong>
    <span>videos_by_category/basic_physics_consistency/motion_3d/082e01d1b61f9360305144d49feea870487197c2560475116c79b8425bdde234.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/09-physics_videos-019c2446b47f1125aaedced3fb5cae96a66e8bd476f3f34642821a8025d569f1.jpg"><source src="../../generated/samples/previews/09-physics_videos-019c2446b47f1125aaedced3fb5cae96a66e8bd476f3f34642821a8025d569f1.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Parabolic Motion</strong>
    <span>videos_by_category/basic_physics_consistency/parabolic_motion/019c2446b47f1125aaedced3fb5cae96a66e8bd476f3f34642821a8025d569f1.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/10-physics_videos-082638c19cc7152742981f17dae9c4c3e8e49f45a4bee420a183ca2f3bece4b1.jpg"><source src="../../generated/samples/previews/10-physics_videos-082638c19cc7152742981f17dae9c4c3e8e49f45a4bee420a183ca2f3bece4b1.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Parabolic Motion With Rotation</strong>
    <span>videos_by_category/basic_physics_consistency/parabolic_motion_with_rotation/082638c19cc7152742981f17dae9c4c3e8e49f45a4bee420a183ca2f3bece4b1.mp4</span>
  </figcaption>
</figure>
</div>

## 可下载产物

- [`summary.json`](../generated/data/summary.json)
- [`all_records.csv`](../generated/data/all_records.csv)
- [`image_records.csv`](../generated/data/image_records.csv)
- [`video_records.csv`](../generated/data/video_records.csv)
- [`physics_records.csv`](../generated/data/physics_records.csv)
