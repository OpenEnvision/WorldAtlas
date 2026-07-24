# Physics

<p class="dimension-lede">The <code>physics</code> dimension covers physical motion realism and simulator-backed physics metrics. It includes motion-dynamics metrics in the default benchmark and simulator-world metrics in the independent physics track.</p>

<div class="dimension-overview dimension-overview--physics">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">Dimension key</div>
  <div class="dimension-hero-value"><code>physics</code></div>
  <div class="dimension-hero-note">Rolls up into <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Tracks</div>
  <div class="dimension-hero-value">3</div>
  <div class="dimension-hero-note">Formal motion, simulator-backed, Physics Judge</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Core question</div>
  <div class="dimension-hero-value dimension-hero-value--compact">Physical realism</div>
  <div class="dimension-hero-note">Do motions and interactions obey plausible physical dynamics?</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Documentation map</p>
<p class="dimension-section-note">Entry points for the three physics evaluation tracks.</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="physics.md">Physics</a></div>
  <p class="dimension-family-note">Dimension entry point: how public benchmark motion metrics, simulator-backed physics, and Physics Judge roll into the <code>physics</code> dimension.</p>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="../physics/index.md">Simulator-Backed Physics</a></div>
  <p class="dimension-family-note">Scope, config, report namespace, and interpretation boundaries for the simulator-backed physics track.</p>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="../physics/workflow.md">Physics Workflow</a></div>
  <p class="dimension-family-note">Manifest construction, prediction layout, run commands, and report outputs.</p>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label"><a href="../physics/metrics.md">Physics Metrics &amp; Contracts</a></div>
  <p class="dimension-family-note">Metric contract, normalization, artifact requirements, and failure semantics.</p>
</div>

</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Scope</p>
<p class="dimension-section-note">Metrics grouped by evaluation track and motion layer.</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">Reference motion dynamics</div>
  <ul class="dimension-family-metrics">
    <li><a href="../pmf.md">pmf</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Motion magnitude and smoothness</div>
  <ul class="dimension-family-metrics">
    <li><a href="../motion_magnitude.md">motion_magnitude</a></li>
    <li><a href="../motion_smoothness.md">motion_smoothness</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Long-sequence dynamics</div>
  <ul class="dimension-family-metrics">
    <li><code>long_sequence_dynamic_degree</code></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Simulator-backed atomic metrics</div>
  <p class="dimension-family-note">Formulas and artifact contracts: <a href="../physics/metrics.md">Physics Metrics &amp; Contracts</a>.</p>
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
  <div class="dimension-family-label">Standalone VLM physics judge</div>
  <ul class="dimension-family-metrics">
    <li><a href="../physics/physics_judge.md">Physics Judge</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">Reading notes</p>
<ul>
<li><code>pmf</code>, <code>motion_magnitude</code>, and <code>motion_smoothness</code> in the default formal suites target video motion dynamics.</li>
<li><code>config/benchmark_physics.yaml</code> is a simulator-backed physics config and should not be mixed directly with the default formal-suite headline.</li>
<li>Simulator-backed formulas, artifact contracts, and failure semantics live in the overview pages <a href="../physics/index.md">Simulator-Backed Physics</a>, <a href="../physics/workflow.md">Physics Workflow</a>, and <a href="../physics/metrics.md">Physics Metrics &amp; Contracts</a>.</li>
<li><a href="../physics/physics_judge.md">Physics Judge</a> belongs to the <code>physics</code> dimension, but it is not a simulator-backed artifact group; it is driven separately by <code>config/benchmark_physics_judge.yaml</code>.</li>
</ul>
</div>

</div>
