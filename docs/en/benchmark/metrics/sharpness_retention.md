# Sharpness Retention

This metric evaluates the stability of fine details by monitoring the evolution of edge gradients. Successful numeric scores are official: they enter the `quality` dimension and the public leaderboard vector. We propose a vectorized Tenengrad method: independently compute the sum of absolute gradients in the horizontal ($G_x$) and vertical ($G_y$) directions, and construct a 2-D sharpness vector to capture texture-direction features:

\[
\mathbf{g}_t
=
\begin{pmatrix}
\sum |G_x| \\
\sum |G_y|
\end{pmatrix}
\tag{6}
\]

where $G_x$ and $G_y$ are Sobel responses ($3\times3$ kernel) on the grayscale frame $I_t$.

A **noise-aware circuit breaker** distinguishes true detail from high-frequency artifacts. Let $n_t$ be the BRISQUE noise score at frame $t$ and $\tau$ a noise threshold. When high noise ($n_t > \tau$) occurs for $5$ consecutive frames, the triggered state is $I_{\mathrm{trig}}=1$. The fusing penalty logic $M$ combines texture similarity with the breaker state (similarity $S(\cdot,\cdot)$ follows the cosine inner product in Eq. (3)):

BRISQUE is undefined for a zero-variance frame. Such a frame receives the configured worst-case `brisque_bad` value and remains in the aggregate, so a collapse to a flat or black frame is penalized rather than silently excluded as a missing score.

\[
M(x_t)
=
\begin{cases}
S(\mathbf{g}_t, \mathbf{g}_1), & I_{\mathrm{trig}} = 0 \;\wedge\; n_t < \tau \\[4pt]
\mathrm{clip}\!\bigl(1 - S(\mathbf{g}_t, \mathbf{g}_1),\, 0,\, 0.2\bigr), & \text{otherwise}
\end{cases}
\tag{11}
\]

An **upper-convex logarithmic transform** $L(\cdot)$ enhances low-to-mid score segments ($x \in [0,1]$, avoiding undefined logarithms and preserving normalization):

\[
L(x)
=
\frac{\ln(1 + k \cdot x)}{\ln(1 + k)},
\quad k > 0
\tag{10}
\]

where $k$ controls the upper-convex degree; $k \in [10, 20]$ is common in vision pipelines. The numerator ensures $x=0 \Rightarrow L(x)=0$; the denominator ensures $x=1 \Rightarrow L(x)=1$.

Temporal aggregation uses normalized exponentially decaying weights $\hat{w}(d)$ (Eq. (5) in the brightness-consistency metric, $d \in [1, T-1]$). The comprehensive score is:

\[
S_{\mathrm{Sharpness}}
=
\sum_{d=1}^{T-1}
\hat{w}(d) \cdot L\!\left(M(x_{d+1})\right)
\tag{12}
\]

where $S(\mathbf{g}_t, \mathbf{g}_1)$ is the cosine similarity between the sharpness vector of frame $t$ and the initial frame. This mechanism distinguishes genuine detail from high-frequency noise and penalizes systematic visual collapse.

<!-- metric-examples:start -->
## Verified Same-Input Examples

Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality.

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/0268f6f099eb__image_static_photorealistic_terrestrial_landscape_0152__input.jpg" alt="Input image: image_static_photorealistic_terrestrial_landscape_0152" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8467</div>
      <h4><code>image_static_photorealistic_terrestrial_landscape_0152</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / terrestrial_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. A desert canyon with rock formations, distant horizon, and sand.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_terrestrial_landscape_0152">
          <source src="../../../assets/metric_examples/curated_20260801/3b795be240b4__image_static_photorealistic_terrestrial_landscape_0152__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9725</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>sharpness_retention</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_terrestrial_landscape_0152">
          <source src="../../../assets/metric_examples/curated_20260801/33b0c6368136__image_static_photorealistic_terrestrial_landscape_0152__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.1258</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>sharpness_retention</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c7cd30d3995e__image_static_photorealistic_passageways_0128__input.jpg" alt="Input image: image_static_photorealistic_passageways_0128" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8771</div>
      <h4><code>image_static_photorealistic_passageways_0128</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan left. A modern hallway with potted plant, artwork, and hallway extension.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">Both first frames passed geometric matching against the displayed input; labels compare this metric only.</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Higher metric score: image_static_photorealistic_passageways_0128">
          <source src="../../../assets/metric_examples/curated_20260801/762e177654f7__image_static_photorealistic_passageways_0128__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9749</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>sharpness_retention</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_photorealistic_passageways_0128">
          <source src="../../../assets/metric_examples/curated_20260801/51080028fdfc__image_static_photorealistic_passageways_0128__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.0978</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>sharpness_retention</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/4369893ac62b__image_static_stylized_living_spaces_0096__input.jpg" alt="Input image: image_static_stylized_living_spaces_0096" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Input verified · same benchmark case · gap 0.8513</div>
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
        <div class="metric-score-pill metric-score-pill--good">Higher metric score · Score 0.9514</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Higher metric score</strong><span>sharpness_retention</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Lower metric score: image_static_stylized_living_spaces_0096">
          <source src="../../../assets/metric_examples/curated_20260801/b8886f3dbd11__image_static_stylized_living_spaces_0096__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">Lower metric score · Score 0.1001</div>
      </div>
      <figcaption class="metric-output-caption"><strong>Lower metric score</strong><span>sharpness_retention</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
