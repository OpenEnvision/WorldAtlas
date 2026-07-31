# Reprojection Error

**reprojection_error** 衡量预测视频在被包装的 DROID-SLAM [[Teed and Deng, 2021](#reference)] 重建路径下，**三维几何与跨帧投影是否自洽**。它只依赖 prediction 本身，不需要 reference video 或相机真值；核心问题是：由生成视频恢复出的相机位姿和稠密深度，能否把场景点重新投影到其他帧的观测位置附近。

与 COLMAP [[Schönberger and Frahm, 2016](#reference-colmap-sfm), [Schönberger et al., 2016](#reference-colmap-mvs)] 等稀疏重建方法主要依赖少量高质量特征点不同，DROID-SLAM 采用**稠密 SLAM** 路径：通过可微分的 Dense Bundle Adjustment（DBA）层，在优化过程中联合细化相机位姿与逐像素深度，使其与光流对应关系保持一致。生成视频的几何错误往往不只出现在少数匹配点附近，也可能表现为背景扭曲、物体局部漂移、深度层次不稳定或跨帧结构无法闭合；稠密 DBA 会利用尽可能多的有效像素和共视点，因此更贴近“整段 rollout 能否被同一个三维解释”的评测目标。

DBA 细化完成后，对有效共视点集合 \(\mathcal{V}\)，重投影误差可写为

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
\tag{1}
\]

其中 \(p^{*}_{ij}\) 是第 \(i\) 帧中的观测像素位置，\(P_{ij}\) 是由位姿与深度恢复的三维点，\(\Pi(\cdot)\) 为投影算子。若生成视频在三维上自洽，同一 scene point 投影回其他帧应接近对应观测；几何漂移、深度不稳定或跨帧结构不一致都会使该距离增大。

在 WorldArena 中，runtime 将整段 prediction 解码为帧序列，并以固定时间步长 4 均匀抽帧运行 DROID-SLAM，以统一约束长视频的显存占用；抽样覆盖整段视频而非只取片段。系统估计轨迹与内部几何，并取有效观测重投影误差 `valid_errors` 的均值作为原始分数 `raw = mean(valid_errors)`。WorldArena 在经验范围 \([0, 1.0719]\) 上反向归一化到 \([0, 1]\)（**归一化分数越高越好**）。若 SLAM 重建失败或未产生有效分数，则返回 `raw = null`、`normalized = null`。DROID-SLAM 计算成功时，该指标在所有 formal suite 中为 **official**，并写入 `consistency_3d_4d` 聚合；它是有用的几何一致性信号，但不能替代 camera error 等几何真值监督，且受固定内参 SLAM 假设影响，跨异构数据解读时应保持谨慎。

## 参考文献 {#reference}

!!! quote "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Zachary Teed and Jia Deng.  
    Advances in Neural Information Processing Systems (NeurIPS), 2021.

<a id="reference-colmap-sfm"></a>

!!! quote "Structure-from-Motion Revisited"

    Johannes L. Schönberger and Jan-Michael Frahm.  
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

<a id="reference-colmap-mvs"></a>

!!! quote "Pixelwise View Selection for Unstructured Multi-View Stereo"

    Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys.  
    European Conference on Computer Vision (ECCV), 2016.

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

<!-- metric-examples:start -->
## 暂无可核验的同 Case 对比

> `artifacts/reports` 中尚未保留该指标可比较的同-case预测对。指标说明继续保留，待至少两份兼容的正式报告齐备后再加入示例。
<!-- metric-examples:end -->
