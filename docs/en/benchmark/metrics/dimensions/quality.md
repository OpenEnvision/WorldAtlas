# Quality

<p class="dimension-lede">The <code>quality</code> dimension covers style, perceptual quality, low-level appearance stability, and reconstruction-style fidelity. It asks whether outputs are visually stable, clear, natural, and, when paired references exist, reconstructively similar.</p>

<div class="dimension-overview dimension-overview--quality">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">Dimension key</div>
  <div class="dimension-hero-value"><code>quality</code></div>
  <div class="dimension-hero-note">Rolls up into <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Metric families</div>
  <div class="dimension-hero-value">4</div>
  <div class="dimension-hero-note">Style, appearance stability, reconstruction, long-sequence quality</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Core question</div>
  <div class="dimension-hero-value dimension-hero-value--compact">Visual stability</div>
  <div class="dimension-hero-note">Clarity, naturalness, and reconstructive fidelity when references exist</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Scope</p>
<p class="dimension-section-note">Metrics grouped by what aspect of visual quality they measure.</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">Style and perceptual quality</div>
  <ul class="dimension-family-metrics">
    <li><a href="../style_consistency.md">style_consistency</a></li>
    <li><a href="../perceptual_quality.md">perceptual_quality</a></li>
    <li><a href="../image_quality.md">image_quality</a></li>
    <li><a href="../hpsv3_norm.md">hpsv3_norm</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Appearance stability</div>
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
  <div class="dimension-family-label">Long-sequence quality</div>
  <ul class="dimension-family-metrics">
    <li><a href="../long_sequence.md">long_sequence_aesthetic_quality</a></li>
    <li><a href="../long_sequence.md">long_sequence_imaging_quality</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">Reading notes</p>
<ul>
<li>The seven complete official <code>image_static</code> quality metrics are <code>sharpness_retention</code>, <code>perceptual_quality</code>, <code>style_consistency</code>, <code>brightness_distribution_consistency</code>, <code>color_temperature_constraint</code>, <code>image_quality</code>, and <code>hpsv3_norm</code>.</li>
<li>All seven enter <code>per_metric_summary</code> and the equal-weight <code>quality</code> dimension total. The legacy aliases <code>brightness_consistency</code> and <code>color_temperature_consistency</code> remain diagnostic.</li>
<li>These low-level metrics are inspired by iWorld-Bench quality slots; they are not reproductions of unpublished iWorld-Bench code.</li>
<li><code>psnr</code> and <code>ssim</code> are paired-frame reconstruction metrics and are better treated as supplementary metrics than replacements for semantic or physical metrics.</li>
<li>Quality metrics alone cannot establish control, memory, or physics correctness.</li>
</ul>
</div>

</div>
