# Collection Details

## Collection Summary

| group     | collection     | modality   |   items |   annotated |   masked |
|:----------|:---------------|:-----------|--------:|------------:|---------:|
| benchmark | dynamic_images | image      |    1628 |        1628 |     1628 |
| benchmark | static_images  | image      |    3372 |        3372 |        0 |
| benchmark | physics_videos | physics    |    1400 |           0 |        0 |
| benchmark | aaa_games      | video      |     465 |         465 |      465 |
| benchmark | dynamic_videos | video      |     788 |           0 |        0 |
| benchmark | static_videos  | video      |    4000 |          31 |       31 |

## Dimension Cardinality

    | collection     |   items |   styles |   scenes |   motion_categories |   physics_dimensions |   avg_size_mb |
|:---------------|--------:|---------:|---------:|--------------------:|---------------------:|--------------:|
| aaa_games      |     465 |        1 |        1 |                   0 |                    0 |      13.9755  |
| dynamic_images |    1628 |        2 |        0 |                   5 |                    0 |       1.41373 |
| dynamic_videos |     788 |        2 |        0 |                   5 |                    0 |      10.7617  |
| physics_videos |    1400 |        2 |       28 |                   0 |                   28 |       7.00255 |
| static_images  |    3372 |        2 |       10 |                   0 |                    0 |       1.08393 |
| static_videos  |    4000 |        2 |       10 |                   0 |                    0 |       7.65937 |

## Mask Summary

| collection     | modality   |   items |   masked |   mask_files |   mask_rate |
|:---------------|:-----------|--------:|---------:|-------------:|------------:|
| dynamic_images | image      |    1628 |     1628 |         2262 |      1      |
| static_images  | image      |    3372 |        0 |            0 |      0      |
| physics_videos | physics    |    1400 |        0 |            0 |      0      |
| aaa_games      | video      |     465 |      465 |          465 |      1      |
| dynamic_videos | video      |     788 |        0 |            0 |      0      |
| static_videos  | video      |    4000 |       31 |           31 |      0.0078 |

## Artifacts

- [`collection_summary.csv`](../generated/data/collection_summary.csv)
- [`annotation_summary.csv`](../generated/data/annotation_summary.csv)
- [`mask_summary.csv`](../generated/data/mask_summary.csv)
