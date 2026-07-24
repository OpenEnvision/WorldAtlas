# Real Time

<p class="dimension-lede"><code>real_time</code> 维度衡量视频世界模型是否能支撑真人交互体验。WorldArena 将其拆分为吞吐、动作延迟与时间标定三个互补问题。</p>

<div class="dimension-overview dimension-overview--real_time">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">维度键</div>
  <div class="dimension-hero-value"><code>real_time</code></div>
  <div class="dimension-hero-note">汇总写入 <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">指标族</div>
  <div class="dimension-hero-value">3</div>
  <div class="dimension-hero-note">吞吐、动作延迟、物理 FPS 标定</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">核心问题</div>
  <div class="dimension-hero-value dimension-hero-value--compact">交互就绪度</div>
  <div class="dimension-hero-note">生成能否跟上播放，并及时响应用户动作</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">范围</p>
<p class="dimension-section-note">关于实时世界生成的三个互补问题。</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">生成吞吐</div>
  <p class="dimension-family-note">模型生成帧的速度，是否至少赶得上用户消费帧的速度。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../realtime_throughput_latency.md#effective-generation-fps">effective_generation_fps</a></li>
    <li><a href="../realtime_throughput_latency.md#realtime-factor">realtime_factor</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">动作延迟</div>
  <p class="dimension-family-note">交互动作变化后，生成画面需要多久才出现可见响应。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../realtime_throughput_latency.md#action-response-latency">action_response_latency</a></li>
    <li><a href="../realtime_throughput_latency.md#latency-budget-pass-rate">latency_budget_pass_rate</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Temporal calibration</div>
  <p class="dimension-family-note">画面运动本身是否拥有物理上合理的时间节奏。</p>
  <ul class="dimension-family-metrics">
    <li><a href="../temporal_calibration.md">physical_fps_error</a></li>
    <li><a href="../temporal_calibration.md">physical_fps_pct_error</a></li>
    <li><a href="../temporal_calibration.md">physical_fps_intra_video_cv</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">指标定义</p>
<p class="dimension-section-note"><code>effective_generation_fps</code> 和 <code>realtime_factor</code> 的详细定义见 <a href="../realtime_throughput_latency.md">实时吞吐与动作延迟</a>。它们衡量生成过程是否能追上播放速度。</p>
<p class="dimension-section-note"><code>action_response_latency</code> 和 <code>latency_budget_pass_rate</code> 的详细定义也见 <a href="../realtime_throughput_latency.md">实时吞吐与动作延迟</a>。它们在存在逐帧 action schedule 时衡量 action 到可见响应的延迟。</p>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">读分注意</p>
<ul>
<li>Throughput 是必要条件，但不是充分条件：模型每秒生成帧数足够高，仍然可能因为 action latency 太高而显得不跟手。</li>
<li>Action latency 只适用于 generation record 中带有 action spec 的模型，例如 Matrix-Game 的 action payload。</li>
<li>YuMe 这类 image/text-conditioned 生成可以报告吞吐指标；如果没有逐帧交互动作表，则 action-latency 指标会标记为 not applicable。</li>
<li>现有 <code>physical_fps_*</code> 指标仍然用于时间节奏标定，但它们不衡量用户输入延迟。</li>
</ul>
</div>

</div>
