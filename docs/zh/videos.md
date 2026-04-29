# 视频分析

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">视频行数</div>
  <div class="metric-value">10,550</div>
  <div class="metric-note">所有已扫描视频</div>
</div>
<div class="metric-card">
  <div class="metric-label">Benchmark 视频</div>
  <div class="metric-value">10,462</div>
  <div class="metric-note">主 benchmark 视频行</div>
</div>
<div class="metric-card">
  <div class="metric-label">已标注</div>
  <div class="metric-value">6,250</div>
  <div class="metric-note">带 annotation bundle 的行</div>
</div>
<div class="metric-card">
  <div class="metric-label">具备精确时长</div>
  <div class="metric-value">10,550</div>
  <div class="metric-note">通过 ffprobe 获取到时长的行</div>
</div>
</div>

## 标注汇总

| collection     |   items |   annotated |   complete_annotations |   annotation_rate |
|:---------------|--------:|------------:|-----------------------:|------------------:|
| static_videos  |    8991 |        5431 |                   5431 |            0.604  |
| dynamic_videos |     971 |         319 |                    319 |            0.3285 |
| aaa_games      |     500 |         500 |                    500 |            1      |
| review_assets  |      88 |           0 |                      0 |            0      |

## 分辨率汇总

| collection     | resolution_label   |   items |
|:---------------|:-------------------|--------:|
| static_videos  | 1280x720           |    7669 |
| static_videos  | 1920x1080          |    1114 |
| aaa_games      | 1920x1080          |     493 |
| dynamic_videos | 640x360            |     283 |
| dynamic_videos | 1280x720           |     256 |
| dynamic_videos | 3840x2160          |     172 |
| static_videos  | 3840x2160          |     124 |
| dynamic_videos | 1920x1080          |     117 |
| review_assets  | 1920x1080          |      65 |
| static_videos  | 1920x960           |      34 |
| dynamic_videos | 7680x4320          |      34 |
| dynamic_videos | 2048x1080          |      29 |
| static_videos  | 7680x4320          |      25 |
| dynamic_videos | 640x338            |      22 |
| dynamic_videos | 426x240            |      22 |

## 视频集合构成

<iframe class="chart-frame" src="../../generated/charts/videos-collection-mix.html" title="Video Collection Mix" loading="lazy"></iframe>

## 时长分布

<iframe class="chart-frame" src="../../generated/charts/videos-duration-mix.html" title="Duration Mix" loading="lazy"></iframe>

## 静态视频层级

<iframe class="chart-frame" src="../../generated/charts/videos-static-video-hierarchy.html" title="Static Video Hierarchy" loading="lazy"></iframe>

## 标注覆盖率

<iframe class="chart-frame" src="../../generated/charts/videos-annotation-coverage.html" title="Annotation Coverage" loading="lazy"></iframe>

## 场景标签分布

<iframe class="chart-frame" src="../../generated/charts/videos-scene-tag-mix.html" title="Scene Tag Mix" loading="lazy"></iframe>

## 相机运动标签

<iframe class="chart-frame" src="../../generated/charts/videos-camera-motion-labels.html" title="Camera Motion Labels" loading="lazy"></iframe>
