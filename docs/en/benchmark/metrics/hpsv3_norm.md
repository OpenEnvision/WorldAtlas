# HPSv3-Norm

HPSv3-Norm measures **human-preference-aligned visual quality** of a generated video using the HPSv3 reward model. It is **prediction-only**: each evaluated frame is scored independently, without pairing against a reference rollout.

`hpsv3_norm` is an official `image_static` quality metric and enters both `per_metric_summary` and the equal-weight seven-metric quality total.

WorldArena uniformly subsamples up to 20 frames from the prediction video. For frame $x_i$, the HPSv3 inferencer outputs a raw reward $r_i$. The per-video mean raw reward is

\[
\bar{r}
=
\frac{1}{N}\sum_{i=1}^{N} r_i
\tag{1}
\]

Let $p_1$ and $p_{99}$ denote fixed normalization percentiles (defaults: $p_1 = 5.21$, $p_{99} = 8.66$). The normalized score on $[0, 100]$ is

\[
S_{\mathrm{HPS}}
=
\mathrm{clip}\!\left(
\frac{\bar{r} - p_1}{p_{99} - p_1} \times 100,\,
0,\,
100
\right)
\tag{2}
\]

WorldArena stores the metric raw value in $[0, 1]$ as $S_{\mathrm{HPS}} / 100$. Higher values indicate stronger human-preference alignment. When the HPSv3 backend or checkpoints are unavailable, the metric is reported as `not_applicable`.

!!! note "Why 1.0000 / 0.0000?"

    These are real evaluation results, not hand-entered placeholders. The gallery intentionally selects verified same-input pairs with the largest normalized score gaps. HPSv3-Norm first averages 20 per-frame raw rewards and then applies Equation (2): a mean at or above $p_{99}=8.66$ is clipped to `1.0000`, while a mean at or below $p_1=5.21$ is clipped to `0.0000`.

    The three displayed pairs have the following raw reward means (higher / lower):

    - Workspaces `0020`: `8.872 / 4.902`
    - Verdant landscape `0125`: `8.863 / 2.239`
    - Stylized living space `0142`: `8.904 / 2.930`

    The raw rewards therefore differ across videos even though percentile normalization maps every displayed pair to the two clipped endpoints.

## References {#reference}

!!! quote "HPSv3 — Human Preference Score"

    Vision-language reward model used to score image quality from human-preference supervision; WorldArena applies linear percentile normalization to per-video mean raw rewards.

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/838324ca602a__image_static_photorealistic_workspaces_0020__input.jpg" alt="Input image: image_static_photorealistic_workspaces_0020" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 1.0000</div>
      <h4><code>image_static_photorealistic_workspaces_0020</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. A futuristic control room with swivel chairs, digital screens, and control panels.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_workspaces_0020">
          <source src="../../../assets/metric_examples/curated_20260801/46ef56e3b4ed__image_static_photorealistic_workspaces_0020__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_workspaces_0020">
          <source src="../../../assets/metric_examples/curated_20260801/6e4b918a70e4__image_static_photorealistic_workspaces_0020__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/12b0daad5e98__image_static_photorealistic_verdant_landscape_0125__input.jpg" alt="Input image: image_static_photorealistic_verdant_landscape_0125" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 1.0000</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0125</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move left. Ancient ruins with side garden, stone bench, and wildflowers.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_verdant_landscape_0125">
          <source src="../../../assets/metric_examples/curated_20260801/e5a2820cbad6__image_static_photorealistic_verdant_landscape_0125__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_verdant_landscape_0125">
          <source src="../../../assets/metric_examples/curated_20260801/4d564f9b2bd0__image_static_photorealistic_verdant_landscape_0125__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/150de03acfae__image_static_stylized_living_spaces_0142__input.jpg" alt="Input image: image_static_stylized_living_spaces_0142" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 1.0000</div>
      <h4><code>image_static_stylized_living_spaces_0142</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera orbit left. A living room with sofa, coffee table, and bookshelves.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_stylized_living_spaces_0142">
          <source src="../../../assets/metric_examples/curated_20260801/6a696264f97e__image_static_stylized_living_spaces_0142__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 1.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0142">
          <source src="../../../assets/metric_examples/curated_20260801/f8fbb4a0c533__image_static_stylized_living_spaces_0142__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0000</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>hpsv3_norm</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
