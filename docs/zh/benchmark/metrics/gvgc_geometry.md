# 极线几何诊断指标

WorldArena 用这一页集中说明 `consistency_3d_4d` 维度下新增的三个 GVGC-style 经典几何指标：

| 指标 | 方向 | 原始值 |
| --- | --- | --- |
| `epipolar_error` | 越小越好 | fundamental-matrix RANSAC 后，inlier Sampson error 的中位数 |
| `epipolar_inlier_rate` | 越大越好 | RANSAC inliers / Lowe-ratio feature matches |
| `feature_match_coverage` | 越大越好 | RANSAC inliers / pair 内较小的 keypoint 数 |

这三个指标是轻量级的几何一致性 probe。它们不替代 DROID-SLAM reprojection、MEt3R 或 reconstruction consistency，而是提供一个透明、便宜、可检查的特征几何信号。

## 衡量目标

核心问题是：相邻生成帧之间，是否能被一个稳定的 two-view epipolar geometry 解释。若同一个静态场景点在两帧中都可见，它们的像素位置应当能由同一个 fundamental matrix \(F\) 约束。几何更稳定的视频，通常会产生更多 epipolar-consistent matches，并且 epipolar residual 更低。

这对 interactive world model / video world model 很有用，因为 3D/4D 不一致经常表现为跨帧对应关系破坏：背景滑动、刚体表面弯曲、重复结构各自漂移，或者相机运动无法由同一个三维场景解释。

## 计算流程

1. 指标接收已加载的 prediction video frames。
2. 每一帧转为灰度图，并按 `max_analysis_width` 做宽度上限缩放。
3. 根据 `pair_stride` 选择 frame pairs，最多评估 `max_pairs` 个 pair。
4. 默认用 SIFT 检测局部特征。如果当前 OpenCV 没有 SIFT，或配置为 `feature_detector: orb`，则使用 ORB。
5. 使用 brute-force matcher 匹配描述子，并用 Lowe ratio test 过滤。
6. 若某个 pair 的 good matches 少于 `min_matches`，该 pair 记为失败。
7. 使用 OpenCV RANSAC 估计 fundamental matrix；若环境支持，会优先使用 `USAC_MAGSAC`。
8. 对 RANSAC inliers 计算 Sampson distance。
9. 对所有 valid pairs 聚合：
    - `epipolar_error`：每个 pair 的 inlier error 中位数，再对 pair 取中位数
    - `epipolar_inlier_rate`：每个 pair 的 inlier rate，再对 pair 取平均
    - `feature_match_coverage`：每个 pair 的 coverage，再对 pair 取平均

如果没有任何 pair 得到足够的有效 inliers，三个指标都会返回 `raw = null`、`normalized = null` 和错误字符串。

## 指标定义

### `epipolar_error` {#epipolar-error}

给定匹配点 \(x_i, x'_i\) 和估计出的 fundamental matrix \(F\)，backend 计算 Sampson error：

\[
d_i =
\frac{|{x'_i}^{T} F x_i|}
{\sqrt{(F x_i)_1^2 + (F x_i)_2^2 + (F^T x'_i)_1^2 + (F^T x'_i)_2^2}}
\]

对每个 valid frame pair，`epipolar_error_pair` 是所有 RANSAC inliers 的 \(d_i\) 中位数。视频级 raw value 是所有 valid pairs 的中位数。

归一化使用 `[0, 5]`，方向为越小越好：

\[
score = 1 - \mathrm{clamp}\left(\frac{raw - 0}{5 - 0}, 0, 1\right)
\]

解读：数值低，说明 inlier correspondences 能较好地落在同一个 epipolar geometry 下；数值高，通常表示几何漂移、two-view consistency 较弱，或 feature matches 不稳定。

### `epipolar_inlier_rate` {#epipolar-inlier-rate}

对每个 valid frame pair：

\[
rate =
\frac{\#\mathrm{RANSAC\ inliers}}{\#\mathrm{good\ feature\ matches}}
\]

视频级 raw value 是所有 valid pairs 的平均值，归一化在 `[0, 1]` 上做恒等映射。

解读：高值说明大部分 descriptor matches 可以被同一个 fundamental matrix 解释；低值说明存在大量相互矛盾的对应关系，常见原因包括几何不一致、强非刚体变化或明显生成伪影。

### `feature_match_coverage` {#feature-match-coverage}

对每个 valid frame pair：

\[
coverage =
\min\left(
\frac{\#\mathrm{RANSAC\ inliers}}
{\min(\#\mathrm{keypoints}_{left}, \#\mathrm{keypoints}_{right})},
1
\right)
\]

视频级 raw value 是所有 valid pairs 的平均值，归一化在 `[0, 1]` 上做恒等映射。

解读：它衡量 epipolar-consistent matches 是否覆盖了足够多可检测结构。这个指标能区分“整幅图像有大量稳定对应点”和“RANSAC 只在一个很小的纹理块上成功”。

## 配置项

默认运行参数位于 `benchmark.metric_runtime.gvgc_geometry`：

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `feature_detector` | `sift` | 特征检测器；也支持 `orb` |
| `max_features` | `2000` | 每帧最多检测的局部特征数量 |
| `ratio_test` | `0.75` | Lowe ratio test 的匹配过滤阈值 |
| `min_matches` | `12` | 尝试 RANSAC 前要求的最少 good matches |
| `max_analysis_width` | `640` | 几何分析时的宽度上限 |
| `ransac_reproj_threshold` | `1.0` | OpenCV fundamental-matrix RANSAC threshold |
| `ransac_confidence` | `0.999` | RANSAC 置信度 |
| `ransac_max_iters` | `10000` | RANSAC 最大迭代数 |
| `pair_stride` | `1` | 被评估 frame pair 的帧间隔 |
| `max_pairs` | `8` | 最多评估的 frame pair 数 |

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 已加载的 prediction video frames |
| 参考输入 | 无 |
| 标注输入 | 无 |
| 后端 | `opencv_epipolar` |
| 聚合状态 | formal 4 suite 中均为 `diagnostic` |
| details 字段 | frame count、pair count、valid/failed pair counts、detector、runtime settings，以及每个 pair 的 match/inlier/error 统计 |
| 失败形式 | prediction 不是 video，或没有 valid epipolar pair 时，返回 `raw = null`、`normalized = null` 和错误字符串 |

## 解读建议

- 这组三个指标最适合有纹理、以静态场景和相机运动为主的视频。
- 对低纹理房间、重复纹理、严重运动模糊、大动态前景、镜头切换，它们的可靠性会下降。
- 单独看 `epipolar_error` 不够：某个 pair 可能只在小区域上拟合得很好，所以要结合 `epipolar_inlier_rate` 和 `feature_match_coverage`。
- coverage 高但 error 高，通常表示可匹配结构不少，但几何约束不一致。
- coverage 低但 error 低，通常表示 RANSAC 找到了很小的干净子集，但证据面太窄，不能作为强几何一致性结论。
