# 视频统计

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">视频条目</div>
  <div class="metric-value">5,253</div>
  <div class="metric-note">All scanned videos</div>
</div>
<div class="metric-card">
  <div class="metric-label">基准视频</div>
  <div class="metric-value">5,253</div>
  <div class="metric-note">Primary benchmark video rows</div>
</div>
<div class="metric-card">
  <div class="metric-label">AAA 游戏视频</div>
  <div class="metric-value">465</div>
  <div class="metric-note">Gameplay subset rows</div>
</div>
<div class="metric-card">
  <div class="metric-label">可读取时长</div>
  <div class="metric-value">5,252</div>
  <div class="metric-note">Rows with ffprobe duration</div>
</div>
</div>

## 标注汇总

| collection     |   items |   annotated |   complete_annotations |   annotation_rate |
|:---------------|--------:|------------:|-----------------------:|------------------:|
| static_videos  |    4000 |          31 |                     31 |            0.0078 |
| dynamic_videos |     788 |           0 |                      0 |            0      |
| aaa_games      |     465 |         465 |                    465 |            1      |

## 分辨率汇总

| collection     | resolution_label   |   items |
|:---------------|:-------------------|--------:|
| static_videos  | 1280x720           |    3527 |
| static_videos  | 1920x1080          |     461 |
| aaa_games      | 1920x1080          |     457 |
| dynamic_videos | 640x360            |     267 |
| dynamic_videos | 1280x720           |     237 |
| dynamic_videos | 1920x1080          |     114 |
| dynamic_videos | 3840x2160          |      71 |
| dynamic_videos | 2048x1080          |      26 |
| dynamic_videos | 426x240            |      22 |
| dynamic_videos | 640x338            |      22 |
| static_videos  | 1920x960           |       6 |
| aaa_games      | 3840x2160          |       6 |
| dynamic_videos | 426x226            |       5 |
| dynamic_videos | 640x310            |       4 |
| dynamic_videos | 2048x872           |       3 |

## 视频样例

<div class="sample-grid sample-grid--video">
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/01-aaa_games-002.jpg"><source src="../../generated/samples/previews/01-aaa_games-002.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/002.mp4</span>
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
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/09-static_videos-01fb867e-ea5b-5bd3-b6cb-75eb76ca105f.jpg"><source src="../../generated/samples/previews/09-static_videos-01fb867e-ea5b-5bd3-b6cb-75eb76ca105f.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Passageways</strong>
    <span>photorealistic/indoor/passageways/0-5s/01fb867e-ea5b-5bd3-b6cb-75eb76ca105f.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/10-static_videos-00c108912e__de5d834d-b513-526d-9e11-20753f027531__supp__4p9s.jpg"><source src="../../generated/samples/previews/10-static_videos-00c108912e__de5d834d-b513-526d-9e11-20753f027531__supp__4p9s.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Workspaces</strong>
    <span>photorealistic/indoor/workspaces/0-5s/00c108912e__de5d834d-b513-526d-9e11-20753f027531__supp__4p9s.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/11-static_videos-016718ab0d__d11903c0-ae54-580e-ba24-c6dd2963ccfc__supp__4p9s.jpg"><source src="../../generated/samples/previews/11-static_videos-016718ab0d__d11903c0-ae54-580e-ba24-c6dd2963ccfc__supp__4p9s.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Aquatic Landscape</strong>
    <span>photorealistic/outdoor/aquatic_landscape/0-5s/016718ab0d__d11903c0-ae54-580e-ba24-c6dd2963ccfc__supp__4p9s.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/12-static_videos-00048c89-aa61-5d72-8ccb-016168c5733c.jpg"><source src="../../generated/samples/previews/12-static_videos-00048c89-aa61-5d72-8ccb-016168c5733c.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / City</strong>
    <span>photorealistic/outdoor/city/0-5s/00048c89-aa61-5d72-8ccb-016168c5733c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/13-static_videos-0006f2e720__989d0856-15b5-5ac3-8f52-29eb47365820__supp__4p9s.jpg"><source src="../../generated/samples/previews/13-static_videos-0006f2e720__989d0856-15b5-5ac3-8f52-29eb47365820__supp__4p9s.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Public Spaces</strong>
    <span>photorealistic/outdoor/public_spaces/0-5s/0006f2e720__989d0856-15b5-5ac3-8f52-29eb47365820__supp__4p9s.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/14-static_videos-0007d92b-c93d-5f56-8c54-b9c208695f64.jpg"><source src="../../generated/samples/previews/14-static_videos-0007d92b-c93d-5f56-8c54-b9c208695f64.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Suburb</strong>
    <span>photorealistic/outdoor/suburb/0-5s/0007d92b-c93d-5f56-8c54-b9c208695f64.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/15-static_videos-000121a0-5d97-5dd8-96a2-0e19299f1bfe.jpg"><source src="../../generated/samples/previews/15-static_videos-000121a0-5d97-5dd8-96a2-0e19299f1bfe.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Terrestrial Landscape</strong>
    <span>photorealistic/outdoor/terrestrial_landscape/0-5s/000121a0-5d97-5dd8-96a2-0e19299f1bfe.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/16-static_videos-0011a54b-799d-5bd6-bcd5-3795f8eb8953__supp__4p9s.jpg"><source src="../../generated/samples/previews/16-static_videos-0011a54b-799d-5bd6-bcd5-3795f8eb8953__supp__4p9s.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Static / Verdant Landscape</strong>
    <span>photorealistic/outdoor/verdant_landscape/0-5s/0011a54b-799d-5bd6-bcd5-3795f8eb8953__supp__4p9s.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/17-aaa_games-003.jpg"><source src="../../generated/samples/previews/17-aaa_games-003.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/003.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/18-aaa_games-004.jpg"><source src="../../generated/samples/previews/18-aaa_games-004.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/004.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/19-aaa_games-005.jpg"><source src="../../generated/samples/previews/19-aaa_games-005.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/005.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/20-aaa_games-006.jpg"><source src="../../generated/samples/previews/20-aaa_games-006.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/006.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/21-aaa_games-007.jpg"><source src="../../generated/samples/previews/21-aaa_games-007.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/007.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/22-aaa_games-010.jpg"><source src="../../generated/samples/previews/22-aaa_games-010.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/010.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/23-aaa_games-011.jpg"><source src="../../generated/samples/previews/23-aaa_games-011.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/011.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../../generated/samples/posters/24-aaa_games-012.jpg"><source src="../../generated/samples/previews/24-aaa_games-012.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Gameplay / Aaa Games</strong>
    <span>videos/012.mp4</span>
  </figcaption>
</figure>
</div>

## Video Collection Mix

<iframe class="chart-frame" src="../../generated/charts/videos-collection-mix.html" title="Video Collection Mix" loading="lazy"></iframe>

## Duration Mix

<iframe class="chart-frame" src="../../generated/charts/videos-duration-mix.html" title="Duration Mix" loading="lazy"></iframe>

## Static Video Hierarchy

<iframe class="chart-frame" src="../../generated/charts/videos-static-video-hierarchy.html" title="Static Video Hierarchy" loading="lazy"></iframe>

## Annotation Coverage

<iframe class="chart-frame" src="../../generated/charts/videos-annotation-coverage.html" title="Annotation Coverage" loading="lazy"></iframe>

## Scene Tag Mix

<iframe class="chart-frame" src="../../generated/charts/videos-scene-tag-mix.html" title="Scene Tag Mix" loading="lazy"></iframe>

## Camera Motion Labels

<iframe class="chart-frame" src="../../generated/charts/videos-camera-motion-labels.html" title="Camera Motion Labels" loading="lazy"></iframe>
