# Collection Details

## Collection Summary

| group     | collection     | modality   |   items |   annotated |   masked |
|:----------|:---------------|:-----------|--------:|------------:|---------:|
| auxiliary | review_assets  | video      |      88 |           0 |        0 |
| benchmark | dynamic_images | image      |    2478 |           0 |      248 |
| benchmark | static_images  | image      |    7040 |           0 |        0 |
| benchmark | aaa_games      | video      |     500 |         500 |      500 |
| benchmark | dynamic_videos | video      |     971 |         319 |      319 |
| benchmark | static_videos  | video      |    8991 |        5431 |     5431 |

## Dimension Cardinality

| collection     |   items |   styles |   scenes |   motion_categories |   avg_size_mb |
|:---------------|--------:|---------:|---------:|--------------------:|--------------:|
| aaa_games      |     500 |        1 |        1 |                   0 |      13.976   |
| dynamic_images |    2478 |        2 |        0 |                   5 |       1.35137 |
| dynamic_videos |     971 |        2 |        0 |                   5 |      27.881   |
| review_assets  |      88 |        0 |        0 |                   0 |      26.5409  |
| static_images  |    7040 |        2 |       10 |                   0 |       1.05243 |
| static_videos  |    8991 |        2 |       10 |                   0 |       7.95079 |

## Mask Summary

| collection     | modality   |   items |   masked |   mask_files |   mask_rate |
|:---------------|:-----------|--------:|---------:|-------------:|------------:|
| dynamic_images | image      |    2478 |      248 |          257 |      0.1001 |
| static_images  | image      |    7040 |        0 |            0 |      0      |
| aaa_games      | video      |     500 |      500 |          500 |      1      |
| dynamic_videos | video      |     971 |      319 |       181000 |      0.3285 |
| review_assets  | video      |      88 |        0 |            0 |      0      |
| static_videos  | video      |    8991 |     5431 |         5431 |      0.604  |

## Artifacts

- [`collection_summary.csv`](../generated/data/collection_summary.csv)
- [`annotation_summary.csv`](../generated/data/annotation_summary.csv)
- [`mask_summary.csv`](../generated/data/mask_summary.csv)
