# Image Quality (MUSIQ)

`image_quality` is an official no-reference `image_static` quality metric. It evaluates prediction frames with the MUSIQ-SPAQ image-quality model and does not require a paired reference rollout.

For each protocol-selected prediction frame $I_i$, MUSIQ produces a native score $q_i$ on its $[0,100]$ scale. WorldArena normalizes and clips each score,

\[
s_i = \mathrm{clip}\!\left(\frac{q_i}{100}, 0, 1\right),
\]

then averages the normalized frame scores:

\[
S_{\mathrm{MUSIQ}} = \frac{1}{N}\sum_{i=1}^{N}s_i.
\]

Higher is better. The resulting value enters `per_metric_summary` and the equal-weight seven-metric quality total. The default backend is `musiq`, with `longer` preprocessing.

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/4369893ac62b__image_static_stylized_living_spaces_0096__input.jpg" alt="Input image: image_static_stylized_living_spaces_0096" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.4837</div>
      <h4><code>image_static_stylized_living_spaces_0096</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A modern living space with hallway, neon lights, and artwork.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0096">
          <source src="../../../assets/metric_examples/curated_20260801/1e7a3127e127__image_static_stylized_living_spaces_0096__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7218</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0096">
          <source src="../../../assets/metric_examples/curated_20260801/b8886f3dbd11__image_static_stylized_living_spaces_0096__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.2381</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/871a90dfcd05__image_static_photorealistic_passageways_0129__input.jpg" alt="Input image: image_static_photorealistic_passageways_0129" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.4037</div>
      <h4><code>image_static_photorealistic_passageways_0129</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. A rustic hallway with wooden floor, window with curtains, and vintage furniture.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0129">
          <source src="../../../assets/metric_examples/curated_20260801/6ae6bcb192f7__image_static_photorealistic_passageways_0129__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7462</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0129">
          <source src="../../../assets/metric_examples/curated_20260801/72ff72478035__image_static_photorealistic_passageways_0129__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3425</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/554f035b9d9c__image_static_photorealistic_passageways_0140__input.jpg" alt="Input image: image_static_photorealistic_passageways_0140" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.4016</div>
      <h4><code>image_static_photorealistic_passageways_0140</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move right. A luxurious hallway with marble columns, doors, and gold-framed mirrors.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0140">
          <source src="../../../assets/metric_examples/curated_20260801/8cdcc534d30e__image_static_photorealistic_passageways_0140__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.7334</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>image_quality</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0140">
          <source src="../../../assets/metric_examples/curated_20260801/5b233f91cddf__image_static_photorealistic_passageways_0140__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.3318</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>image_quality</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
