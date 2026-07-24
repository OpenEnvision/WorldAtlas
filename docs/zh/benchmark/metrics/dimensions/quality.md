# Quality

<p class="dimension-lede"><code>quality</code> 维度覆盖风格、感知质量、低层外观稳定性与重建式保真。它关注输出是否视觉上稳定、清晰、自然，以及在 paired reference 存在时是否有较好的重建相似性。</p>

<div class="dimension-overview dimension-overview--quality">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">维度键</div>
  <div class="dimension-hero-value"><code>quality</code></div>
  <div class="dimension-hero-note">汇总写入 <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">指标族</div>
  <div class="dimension-hero-value">4</div>
  <div class="dimension-hero-note">风格、外观稳定性、重建式保真、长序列质量</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">核心问题</div>
  <div class="dimension-hero-value dimension-hero-value--compact">视觉稳定性</div>
  <div class="dimension-hero-note">清晰度、自然度，以及有参考时的重建相似性</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">覆盖范围</p>
<p class="dimension-section-note">按视觉质量的不同层面组织指标。</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">风格与感知质量</div>
  <ul class="dimension-family-metrics">
    <li><a href="../style_consistency.md">style_consistency</a></li>
    <li><a href="../perceptual_quality.md">perceptual_quality</a></li>
    <li><a href="../image_quality.md">image_quality</a></li>
    <li><a href="../hpsv3_norm.md">hpsv3_norm</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">外观稳定性</div>
  <ul class="dimension-family-metrics">
    <li><a href="../brightness_consistency.md">brightness_distribution_consistency</a></li>
    <li><a href="../color_temperature_consistency.md">color_temperature_constraint</a></li>
    <li><a href="../sharpness_retention.md">sharpness_retention</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Paired reconstruction</div>
  <ul class="dimension-family-metrics">
    <li><a href="../psnr.md">psnr</a></li>
    <li><a href="../ssim.md">ssim</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Long-Sequence quality</div>
  <ul class="dimension-family-metrics">
    <li><a href="../long_sequence.md">long_sequence_aesthetic_quality</a></li>
    <li><a href="../long_sequence.md">long_sequence_imaging_quality</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">解读重点</p>
<ul>
<li>七个完整的正式 <code>image_static</code> quality 指标为：<code>sharpness_retention</code>、<code>perceptual_quality</code>、<code>style_consistency</code>、<code>brightness_distribution_consistency</code>、<code>color_temperature_constraint</code>、<code>image_quality</code>、<code>hpsv3_norm</code>。</li>
<li>七项全部进入 <code>per_metric_summary</code>，并等权计入 <code>quality</code> 维度总分；旧别名 <code>brightness_consistency</code> 与 <code>color_temperature_consistency</code> 仍保留为 diagnostic。</li>
<li>这些低层指标受 iWorld-Bench quality 槽位启发，但不是对未公开 iWorld-Bench 官方代码的复现。</li>
<li><code>psnr</code> / <code>ssim</code> 是 paired-frame 重建式指标，更适合作为补充指标，不应替代语义或物理指标。</li>
<li>quality 指标无法单独说明 control、memory、physics 是否正确。</li>
</ul>
</div>

</div>
