# Physics

<p class="dimension-lede"><code>physics</code> 维度覆盖运动物理真实性与 simulator-backed physics 指标。它包含默认 benchmark 中的运动动态指标，也包含独立 physics track 中由 simulator world artifact 支撑的指标。</p>

<div class="dimension-overview dimension-overview--physics">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">维度键</div>
  <div class="dimension-hero-value"><code>physics</code></div>
  <div class="dimension-hero-note">汇总写入 <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">评测轨道</div>
  <div class="dimension-hero-value">3</div>
  <div class="dimension-hero-note">Formal 运动、simulator-backed、Physics Judge</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">核心问题</div>
  <div class="dimension-hero-value dimension-hero-value--compact">物理真实性</div>
  <div class="dimension-hero-note">运动与交互是否符合合理的物理动态</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">文档地图</p>
<p class="dimension-section-note">三条 physics 评测轨道的入口页面。</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="physics.md">Physics</a></div>
  <p class="dimension-family-note">当前维度入口：说明 public benchmark 运动指标、simulator-backed physics、Physics Judge 三者如何共同归入 <code>physics</code> 维度。</p>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="../physics/index.md">Simulator-Backed Physics</a></div>
  <p class="dimension-family-note">simulator-backed physics track 的范围、配置、report namespace 与解释边界。</p>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="../physics/workflow.md">Physics Workflow</a></div>
  <p class="dimension-family-note">manifest 构建、预测目录、运行命令和报告产物。</p>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="../physics/metrics.md">Physics Metrics &amp; Contracts</a></div>
  <p class="dimension-family-note">指标契约、归一化、artifact 要求和 failure semantics。</p>
</div>

</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">覆盖范围</p>
<p class="dimension-section-note">按评测轨道与运动层面组织指标。</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">参考运动动态</div>
  <ul class="dimension-family-metrics">
    <li><a href="../pmf.md">pmf</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">运动幅度与平滑</div>
  <ul class="dimension-family-metrics">
    <li><a href="../motion_magnitude.md">motion_magnitude</a></li>
    <li><a href="../motion_smoothness.md">motion_smoothness</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Long-Sequence 动态指标</div>
  <ul class="dimension-family-metrics">
    <li><code>long_sequence_dynamic_degree</code></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Simulator-backed 原子指标</div>
  <p class="dimension-family-note">公式与 artifact 契约见 <a href="../physics/metrics.md">Physics Metrics &amp; Contracts</a>。</p>
  <ul class="dimension-family-metrics">
    <li><code>mse</code></li>
    <li><code>st_iou</code></li>
    <li><code>s_iou</code></li>
    <li><code>ws_iou</code></li>
    <li><code>abs_rel</code></li>
    <li><code>rmse</code></li>
    <li><code>delta1</code></li>
    <li><code>delta2</code></li>
    <li><code>delta3</code></li>
    <li><code>depth_warp_l1</code></li>
    <li><code>depth_warp_charb</code></li>
    <li><code>rgb_warp_charb</code></li>
    <li><code>epe</code></li>
    <li><code>fl_all</code></li>
    <li><code>one_px_out</code></li>
    <li><code>chamfer_4d</code></li>
    <li><code>worldline_l2_error</code></li>
    <li><code>trajectory_rmse</code></li>
    <li><code>final_state_accuracy</code></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">独立 VLM physics judge</div>
  <ul class="dimension-family-metrics">
    <li><a href="../physics/physics_judge.md">Physics Judge</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">解读重点</p>
<ul>
<li>默认 formal suite 中的 <code>pmf</code>、<code>motion_magnitude</code>、<code>motion_smoothness</code> 面向视频运动动态。</li>
<li><code>config/benchmark_physics.yaml</code> 是 simulator-backed physics 配置，不应和默认 <code>config/benchmark.yaml</code> 的 formal suite headline 直接混合。</li>
<li>simulator-backed 指标的数学公式、artifact 契约和 failure semantics 以总览中的 <a href="../physics/index.md">Simulator-Backed Physics</a>、<a href="../physics/workflow.md">Physics Workflow</a> 和 <a href="../physics/metrics.md">Physics Metrics &amp; Contracts</a> 为准。</li>
<li><a href="../physics/physics_judge.md">Physics Judge</a> 属于 <code>physics</code> 维度，但它不是 simulator-backed artifact group；它由 <code>config/benchmark_physics_judge.yaml</code> 单独驱动。</li>
</ul>
</div>

</div>
