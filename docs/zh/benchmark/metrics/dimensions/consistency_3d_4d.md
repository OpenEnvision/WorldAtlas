# 3D/4D 一致性

`consistency_3d_4d` 维度用于评估生成 rollout 是否具备**可被稳定三维/四维结构解释**的时空几何一致性。与仅考察单帧视觉合理性或低层外观稳定性的质量指标不同，本维度关注相机运动、深度层次、跨帧对应、新显露区域与运动场在整段视频上能否被同一几何假设同时解释。该维度的汇总结果写入 `per_dimension_summary.json`，核心问题是生成视频是否像在三维世界中沿连续视点演化。

### 指标族与几何语义

本维度覆盖六类几何读数，组织顺序从可见区域与新显露区域，到局部极线约束与全局重建，再到深度结构和短时运动场。

在新显露区域方面，[`novel_region_fidelity`](../novel_region_fidelity.md) 检查参考相机路径暴露出第一帧不可见区域时，预测是否保持 masked 外观保真。在多视角与 3D 表征方面，[`met3r_consistency`](../met3r_consistency.md) 衡量锚点视角间的表征一致性，[`reconstruction_consistency`](../reconstruction_consistency.md) 则检查预测能否被 3D Gaussian 重建并沿视点路径重渲染。

经典极线几何由 [`epipolar_error`](../gvgc_geometry.md#epipolar-error)、[`epipolar_inlier_rate`](../gvgc_geometry.md#epipolar-inlier-rate) 和 [`feature_match_coverage`](../gvgc_geometry.md#feature-match-coverage) 读出，用于判断相邻帧是否满足稳定的 fundamental-matrix / Sampson 极线解释。相机与重投影方面，[`reprojection_error`](../reprojection_error.md) 检查稠密 SLAM 恢复出的位姿与深度能否将共视点一致地重投影回图像平面。

深度结构由 [`depth_accuracy`](../depth_accuracy.md) 和 [`depth_collision`](../depth_collision.md) 共同刻画：前者关注相对深度排序是否保留，后者关注前向视廊是否出现异常近距几何。运动场方面，[`optical_flow_aepe`](../optical_flow_aepe.md) 检查相邻帧局部外观在正反向光流往返下是否保持短时自洽。

### 形式化读数

设对某一样本、某一 formal suite，在 `consistency_3d_4d` 下成功计算且标记为 **official** 的指标集合为 $\mathcal{M}_{\mathrm{off}}$。每个指标 $m \in \mathcal{M}_{\mathrm{off}}$ 经后端归一化后给出 $\tilde{s}_m \in [0,1]$，且统一约定**越高越好**。WorldArena 将该维度的 suite 级汇总写为 official 指标归一化分数的算术平均：

\[
S_{\mathrm{3D/4D}}
=
\frac{1}{|\mathcal{M}_{\mathrm{off}}|}
\sum_{m \in \mathcal{M}_{\mathrm{off}}}
\tilde{s}_m
\tag{1}
\]

式 (1) 仅聚合 **official** 读数；非 official 指标提供互补几何探针，但不进入 `per_dimension_summary.json`。若某 official 指标因缺输入、重建失败或监督门槛未满足而返回 `normalized = null`，则该指标不参与式 (1)，且不会被自动记为 $0$。

从几何约束强度看，极线三元组提供**最轻量**的双视角一致性信号，适合纹理充足、以静态场景与相机运动为主的 rollout；`reprojection_error` 与重建类指标则要求预测视频在**整段轨迹**上支持稠密三维解释；`novel_region_fidelity` 进一步把一致性检验扩展到**随相机运动新进入视野**的区域，而非仅比较持续可见区域。

`consistency_3d_4d` 衡量的是**生成视频内部的几何自洽性**，而非指令跟随、语义正确性或审美质量。它不替代 [`action_control`](action_control.md) 中的相机/运动控制读数，也不等价于 [`quality`](quality.md) 中的风格与感知质量指标。报告时应将 $S_{\mathrm{3D/4D}}$ 与动作控制、质量及物理维度并列呈现：高分表示生成结果较容易被统一的三维/四维结构解释，但并不保证 prompt 被正确执行，也不保证物体行为符合物理直觉。
