# Long Sequence (Memory)

<p class="dimension-lede">The <code>long_sequence</code> dimension covers state memory, appearance preservation, and long-rollout stability. It asks whether the model preserves seen regions, object presence, revisit behavior, and long-horizon motion or quality over time.</p>

<div class="dimension-overview dimension-overview--long_sequence">

<div class="dimension-hero-grid">
<div class="dimension-hero-card dimension-hero-card--primary">
  <div class="dimension-hero-label">Dimension key</div>
  <div class="dimension-hero-value"><code>long_sequence</code></div>
  <div class="dimension-hero-note">Rolls up into <code>per_dimension_summary.json</code></div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Metric families</div>
  <div class="dimension-hero-value">3</div>
  <div class="dimension-hero-note">Revisit stability, persistence, long-horizon motion</div>
</div>
<div class="dimension-hero-card">
  <div class="dimension-hero-label">Core question</div>
  <div class="dimension-hero-value dimension-hero-value--compact">State memory</div>
  <div class="dimension-hero-note">Does the model remember what it already generated?</div>
</div>
</div>

<div class="dimension-section">
<p class="dimension-section-label">Scope</p>
<p class="dimension-section-note">Metrics grouped by what kind of long-horizon memory they test.</p>

<div class="dimension-family-grid">

<div class="dimension-family-card">
  <div class="dimension-family-label">Revisit stability</div>
  <ul class="dimension-family-metrics">
    <li><a href="../memory_revisit_consistency.md">memory_revisit_consistency</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Seen regions and object persistence</div>
  <ul class="dimension-family-metrics">
    <li><a href="../seen_region_preservation.md">seen_region_preservation</a></li>
    <li><a href="../object_permanence.md">object_permanence</a></li>
  </ul>
</div>

<div class="dimension-family-card">
  <div class="dimension-family-label">Long-sequence backend metrics</div>
  <ul class="dimension-family-metrics">
    <li><a href="../long_sequence.md">long_sequence_motion_smoothness</a></li>
  </ul>
</div>

</div>
</div>

<div class="dimension-notes">
<p class="dimension-notes-label">Reading notes</p>
<ul>
<li>These metrics mainly answer whether the model remembers what it has already generated.</li>
<li><code>seen_region_preservation</code> and <code>object_permanence</code> are strongest when reference video and masks are available; otherwise they are usually <code>not_applicable</code> or non-official.</li>
<li>Bundled Long-Sequence metrics should be read alongside formal-suite metrics, not as a standalone headline score.</li>
</ul>
</div>

</div>
