# Reprojection Error

`reprojection_error` 用来衡量预测视频在被包装的 DROID-SLAM [[Teed and Deng, 2021](#reference)] 重建路径下，三维几何与跨帧投影是否自洽。
它关注的是：由生成视频自身恢复出的相机位姿和稠密深度，在重新投影回其他帧时，能否落在与实际观测一致的位置。
误差越小，说明视频中的空间结构、相机运动和跨帧对应关系越容易被同一个三维重建解释。

## 这个指标测什么

- 预测结果在 SLAM backend 下是否容易被一致地重建和重投影。
- 生成视频中场景几何、相机运动和跨帧对应点的空间一致性。

## 原理说明

DROID-SLAM 的一个关键优势是它采用稠密 SLAM 路径。
与 COLMAP [[Schönberger and Frahm, 2016](#reference-colmap-sfm), [Schönberger et al., 2016](#reference-colmap-mvs)] 这类稀疏重建方法不同，稀疏方法通常会选择质量较高的特征匹配点，并丢弃大量不稳定或低置信的区域。
DROID-SLAM 会使用可微分的 Dense Bundle Adjustment（DBA）层，在优化过程中持续联合细化相机位姿和稠密的逐像素深度估计，使它们与当前光流对应关系保持一致。

这种稠密重建方式更适合用来评估生成视频的 3D consistency。
原因是生成视频的几何错误不一定只出现在少量特征点附近，也可能表现为背景扭曲、物体局部漂移、深度层次不稳定、跨帧结构不能闭合等全局或局部问题。
稠密 DBA 会利用尽可能多的有效像素和共视点，而不是只依赖部分“好匹配”，因此更贴近“整幅画面在三维空间中是否协调一致”的评测目标。

## 具体算法

1. WorldArena 会把整段预测视频解码成 PNG 帧。
2. 被包装的第三方指标会在这些帧上运行 `DROID-SLAM`，估计相机轨迹、稠密深度和内部几何结构。
3. DROID-SLAM 通过 DBA 层细化位姿和深度后，对共视区域内的有效点计算重投影误差。
4. 在 DROID-SLAM 完成整段视频的优化后，它会返回一组有效观测点的重投影误差 `valid_errors`；这些误差表示重建出的三维点重新投影到图像平面后，与对应观测像素位置之间的距离。
5. 原始指标值定义为 `raw = mean(valid_errors)`。
6. WorldArena 会按经验边界对这个标量做归一化：
    - `empirical_min = 0.0`
    - `empirical_max = 1.0719`
    - 越小越好
7. 因而归一化分数本质上等价于 `1 - clamp((raw - 0.0) / (1.0719 - 0.0), 0, 1)`。

## 误差公式

DBA 细化完成后，重投影误差可以写成：

!!! info "核心公式"

    \[
    e_{\mathrm{reproj}}
    =
    \frac{1}{|\mathcal{V}|}
    \sum_{(i,j)\in\mathcal{V}}
    \left\lVert
        p^{*}_{ij}
        -
        \Pi(P_{ij})
    \right\rVert_2
    \]

其中，\(\mathcal{V}\) 表示有效的共视点集合；\(p^{*}_{ij}\) 是第 \(i\) 帧中与第 \(j\) 个三维点对应的观测像素位置；\(P_{ij}\) 是由 DBA 细化后的深度和相机位姿恢复出的三维点；\(\Pi(\cdot)\) 表示用当前相机模型把三维点投影回图像平面；\(\lVert\cdot\rVert_2\) 表示二维像素位置之间的欧氏距离。

直观地说，如果生成视频在三维上自洽，那么同一个场景点从一帧恢复到三维空间后，再投影到另一帧，应当落在对应的观测位置附近。
如果画面中存在明显的几何漂移、深度不稳定或跨帧结构不一致，投影位置和观测位置之间的距离会变大，`reprojection_error` 也会升高。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 无 |
| 标注输入 | 无 |
| 原始输出 | 单个 reprojection error 标量 |
| 归一化输出 | 单个 `[0, 1]` 标量，越高越好 |
| 细节字段 | `frame_count`、`native_score` |
| 失败形式 | 如果 SLAM 重建失败或没有得到有效分数，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 参考文献 {#reference}

!!! quote "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Zachary Teed and Jia Deng.  
    Advances in Neural Information Processing Systems (NeurIPS), 2021.

??? note "BibTeX"

    ```bibtex
    @article{teed2021droid,
        title = {Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras},
        author = {Teed, Zachary and Deng, Jia},
        journal = {Advances in neural information processing systems},
        volume = {34},
        pages = {16558--16569},
        year = {2021},
    }
    ```


<a id="reference-colmap-sfm"></a>

!!! quote "Structure-from-Motion Revisited"

    Johannes L. Schönberger and Jan-Michael Frahm.  
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.


<a id="reference-colmap-mvs"></a>

!!! quote "Pixelwise View Selection for Unstructured Multi-View Stereo"

    Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys.  
    European Conference on Computer Vision (ECCV), 2016.
