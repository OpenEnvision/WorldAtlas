# Real Time

<p class="dimension-lede">The <code>real_time</code> dimension covers whether a video world model can support an interactive human experience. WorldArena separates this into throughput, action latency, and temporal calibration.</p>

<div class="dimension-overview dimension-overview--real_time">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">Dimension key</div>
  <div class="dimension-hero-value"><code>real_time</code></div>
  <div class="dimension-hero-note">Rolls up into <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Metric families</div>
  <div class="dimension-hero-value">3</div>
  <div class="dimension-hero-note">Throughput, action latency, physical-FPS calibration</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Core question</div>
  <div class="dimension-hero-value dimension-hero-value--compact">Interactive readiness</div>
  <div class="dimension-hero-note">Can generation keep up and respond to user actions in time?</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Scope</p>
<p class="dimension-section-note">Three complementary questions about real-time world generation.</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">Generation throughput</div>
  <p class="dimension-family-note">Can the model generate frames at least as fast as users consume them?</p>
  <ul class="dimension-family-metrics">
    <li><a href="../realtime_throughput_latency.md#effective-generation-fps">effective_generation_fps</a></li>
    <li><a href="../realtime_throughput_latency.md#realtime-factor">realtime_factor</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Action latency</div>
  <p class="dimension-family-note">After an action changes, how long until generated frames visibly respond?</p>
  <ul class="dimension-family-metrics">
    <li><a href="../realtime_throughput_latency.md#action-response-latency">action_response_latency</a></li>
    <li><a href="../realtime_throughput_latency.md#latency-budget-pass-rate">latency_budget_pass_rate</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Temporal calibration</div>
  <p class="dimension-family-note">Does visual motion follow a physically plausible clock?</p>
  <ul class="dimension-family-metrics">
    <li><a href="../temporal_calibration.md">physical_fps_error</a></li>
    <li><a href="../temporal_calibration.md">physical_fps_pct_error</a></li>
    <li><a href="../temporal_calibration.md">physical_fps_intra_video_cv</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Metric definitions</p>
<p class="dimension-section-note"><code>effective_generation_fps</code> and <code>realtime_factor</code> are detailed in <a href="../realtime_throughput_latency.md">Realtime Throughput and Latency</a>. They measure whether generation keeps up with playback.</p>
<p class="dimension-section-note"><code>action_response_latency</code> and <code>latency_budget_pass_rate</code> are also detailed in <a href="../realtime_throughput_latency.md">Realtime Throughput and Latency</a>. They measure action-to-visible-response delay when frame-level action schedules are available.</p>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">Reading notes</p>
<ul>
<li>Throughput is necessary but not sufficient: a model can generate enough frames per second while still feeling unresponsive if action latency is high.</li>
<li>Action latency is only applicable when generation records include an action spec, such as Matrix-Game action payloads.</li>
<li>YuMe-style image/text-conditioned generation can report throughput metrics, but action-latency metrics are not applicable unless a frame-level interaction schedule is available.</li>
<li>The existing <code>physical_fps_*</code> metrics remain useful for timing calibration, but they do not measure user input latency.</li>
</ul>
</div>

</div>
