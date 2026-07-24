# 3D/4D 一致性

<p class="dimension-lede"><code>consistency_3d_4d</code> 维度用于评估生成 rollout 是否具备<strong>可被稳定三维/四维结构解释</strong>的时空几何一致性。与仅考察单帧视觉合理性或低层外观稳定性的质量指标不同，本维度关注相机运动、深度层次、跨帧对应、新显露区域与运动场在整段视频上能否被同一几何假设同时解释。</p>

<div class="dimension-overview dimension-overview--consistency_3d_4d">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">维度键</div>
  <div class="dimension-hero-value"><code>consistency_3d_4d</code></div>
  <div class="dimension-hero-note">汇总写入 <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">指标族</div>
  <div class="dimension-hero-value">6</div>
  <div class="dimension-hero-note">新显露区域、多视角、极线、重投影、深度、运动场</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">核心问题</div>
  <div class="dimension-hero-value dimension-hero-value--compact">几何自洽性</div>
  <div class="dimension-hero-note">视频是否像在三维世界中沿连续视点演化</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">指标族与几何语义</p>
<p class="dimension-section-note">从可见区域到新显露区域、从局部极线约束到全局重建、从深度结构到短时运动场组织读数。</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">新显露区域</div>
  <p class="dimension-family-note">参考相机路径暴露第一帧不可见区域时，预测是否保持 masked 外观保真。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../novel_region_fidelity.md">novel_region_fidelity</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">多视角 / 3D 表征</div>
  <p class="dimension-family-note">锚点视角间的表征一致性，以及预测能否被 3D Gaussian 重建并沿视点路径重渲染。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../met3r_consistency.md">met3r_consistency</a></li>
    <li><a href="../reconstruction_consistency.md">reconstruction_consistency</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">经典极线几何</div>
  <p class="dimension-family-note">相邻帧是否满足稳定的 fundamental-matrix / Sampson 极线解释。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../gvgc_geometry.md#epipolar-error">epipolar_error</a></li>
    <li><a href="../gvgc_geometry.md#epipolar-inlier-rate">epipolar_inlier_rate</a></li>
    <li><a href="../gvgc_geometry.md#feature-match-coverage">feature_match_coverage</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">相机与重投影</div>
  <p class="dimension-family-note">稠密 SLAM 恢复出的位姿与深度能否将共视点一致地重投影回图像平面。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../reprojection_error.md">reprojection_error</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">深度结构</div>
  <p class="dimension-family-note">相对深度排序是否保留，以及前向视廊是否出现异常近距几何。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../depth_accuracy.md">depth_accuracy</a></li>
    <li><a href="../depth_collision.md">depth_collision</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">运动场</div>
  <p class="dimension-family-note">相邻帧局部外观在正反向光流往返下是否短时自洽。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../optical_flow_aepe.md">optical_flow_aepe</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-body">

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

### Official 与 Diagnostic 分工

在当前公开 benchmark 面上，`novel_region_fidelity`、`reprojection_error`、`optical_flow_aepe` 与 `met3r_consistency` 在计算成功时进入 **official** 聚合；`depth_accuracy`、`depth_collision`、`reconstruction_consistency` 以及 GVGC-style 极线指标当前单独报告，不进 headline 聚合。该分工反映了不同指标在监督形态、失败模式与 suite eligibility 上的差异。MEt3R 的空 overlap mask 会明确计为最差分（归一化 `0`）；视频不可读和 runtime 错误才作为失败并从均值中排除。极线指标依赖 OpenCV 特征匹配，在弱纹理或强非刚性运动场景中可能返回 `not_applicable`；重建类指标则对完整视频解码与第三方 runtime 可用性更敏感。

### 解读边界

`consistency_3d_4d` 衡量的是**生成视频内部的几何自洽性**，而非指令跟随、语义正确性或审美质量。它不替代 [`action_control`](action_control.md) 中的相机/运动控制读数，也不等价于 [`quality`](quality.md) 中的风格与感知质量指标。报告时应将 $S_{\mathrm{3D/4D}}$ 与动作控制、质量及物理维度并列呈现：高分表示生成结果较容易被统一的三维/四维结构解释，但并不保证 prompt 被正确执行，也不保证物体行为符合物理直觉。

</div>

</div>
