# Long Sequence（Memory）

<p class="dimension-lede"><code>long_sequence</code> 维度覆盖状态记忆、外观保持与长时序 rollout 稳定性。它关注模型是否能在长视频中维持已见区域、物体存在性、回访行为，以及长时运动或画质表现。</p>

<div class="dimension-overview dimension-overview--long_sequence">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">维度键</div>
  <div class="dimension-hero-value"><code>long_sequence</code></div>
  <div class="dimension-hero-note">汇总写入 <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">指标族</div>
  <div class="dimension-hero-value">3</div>
  <div class="dimension-hero-note">回访稳定性、持久性、长时运动</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">核心问题</div>
  <div class="dimension-hero-value dimension-hero-value--compact">状态记忆</div>
  <div class="dimension-hero-note">模型是否记得之前生成过什么</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">覆盖范围</p>
<p class="dimension-section-note">按长时记忆的不同层面组织指标。</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">回访稳定性</div>
  <ul class="dimension-family-metrics">
    <li><a href="../memory_revisit_consistency.md">memory_revisit_consistency</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">已见区域与物体持久性</div>
  <ul class="dimension-family-metrics">
    <li><a href="../seen_region_preservation.md">seen_region_preservation</a></li>
    <li><a href="../object_permanence.md">object_permanence</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Long-Sequence backend 指标</div>
  <ul class="dimension-family-metrics">
    <li><a href="../long_sequence.md">long_sequence_motion_smoothness</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">解读重点</p>
<ul>
<li>这些指标主要回答“模型是否记得之前生成过什么”。</li>
<li><code>seen_region_preservation</code> 与 <code>object_permanence</code> 在有 reference video / mask 时更强；监督缺失时通常为 <code>not_applicable</code> 或非 official。</li>
<li>bundled Long-Sequence 指标适合和 formal suite 指标一起看，不单独构成 headline 总分。</li>
</ul>
</div>

</div>
