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
## Representative Same-Case Comparisons
Each exhibit first shows the input image and the exact displayed prompt for the case, then compares generated outputs. Paired exhibits require the same input image and the same prompt text, with one high-scoring and one low-scoring model output. Metrics without such a cross-model pair in the current formal snapshot are separated as high/low extrema rather than presented as same-case comparisons.
Machine-readable paired export: `docs/assets/metric_examples/paired_manifest.csv`.
### `sharpness_retention`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/564b5211b4ae__image_static_photorealistic_passageways_0077.jpg" alt="Input image: image_static_photorealistic_passageways_0077" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · same input image + same prompt · score gap 1.0000</div>
      <h4><code>image_static_photorealistic_passageways_0077</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pan left. A japanese pathway with torii gates, stone lantern, and fox statues. Camera move left. A japanese garden with small pond, stone bridge, and red foliage. Camera pull out. A japanese garden with tea house, bamboo grove, and garden plants.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_passageways_0077">
      <source src="../../../assets/metric_examples/641521aff88e__cosmos3-nano-image-static__image_static_photorealistic_passageways_0077.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_passageways_0077">
      <source src="../../../assets/metric_examples/4e91b9f3e95c__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_passageways_0077.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2f79eec6bacb__image_static_photorealistic_public_spaces_0037.jpg" alt="Input image: image_static_photorealistic_public_spaces_0037" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · same input image + same prompt · score gap 1.0000</div>
      <h4><code>image_static_photorealistic_public_spaces_0037</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move right. Modern buildings with street and pedestrians.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_public_spaces_0037">
      <source src="../../../assets/metric_examples/7e1b20e236f8__cosmos3-nano-image-static__image_static_photorealistic_public_spaces_0037.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_public_spaces_0037">
      <source src="../../../assets/metric_examples/5b5643c5c871__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_public_spaces_0037.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/45fb3a6472c9__image_static_photorealistic_suburb_0012.jpg" alt="Input image: image_static_photorealistic_suburb_0012" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · same input image + same prompt · score gap 1.0000</div>
      <h4><code>image_static_photorealistic_suburb_0012</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / suburb</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera pull out. A suburban landscape with city skyline, green hills, and road. Camera move right. A suburban landscape with road, residential area, and green hills. Camera pan right. A residential area with park, playground, and green hills.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_photorealistic_suburb_0012">
      <source src="../../../assets/metric_examples/3c1ba8c5ffad__wan22-ti2v-5b-image-static-plan-missing__image_static_photorealistic_suburb_0012.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_photorealistic_suburb_0012">
      <source src="../../../assets/metric_examples/9f1698417770__dreamx_world__image_static_photorealistic_suburb_0012.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f116c80bb018__image_static_stylized_dining_spaces_0027.jpg" alt="Input image: image_static_stylized_dining_spaces_0027" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · same input image + same prompt · score gap 1.0000</div>
      <h4><code>image_static_stylized_dining_spaces_0027</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera move left. A modern kitchen with dining table, chairs, and greenery outside. Camera pan left. A modern kitchen with sofa, coffee table, and living area. Camera pull out. A modern kitchen with hallway, artwork, and indoor plants.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_dining_spaces_0027">
      <source src="../../../assets/metric_examples/166bdb8a9374__astra__image_static_stylized_dining_spaces_0027.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_dining_spaces_0027">
      <source src="../../../assets/metric_examples/8119a3b3dbfc__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_dining_spaces_0027.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2c0214795c05__image_static_stylized_passageways_0073.png" alt="Input image: image_static_stylized_passageways_0073" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · same input image + same prompt · score gap 1.0000</div>
      <h4><code>image_static_stylized_passageways_0073</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit left. A stylized hallway with console table, circular mirror, and lamps.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_passageways_0073">
      <source src="../../../assets/metric_examples/95936cc0040a__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_passageways_0073.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_passageways_0073">
      <source src="../../../assets/metric_examples/da49df4a04bd__warp-as-history-image-static-8gpu-20260612_152000__image_static_stylized_passageways_0073.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/38fa6cdab0e6__image_static_stylized_public_spaces_0033.jpg" alt="Input image: image_static_stylized_public_spaces_0033" loading="lazy" decoding="async">
      <figcaption>Input image</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · same input image + same prompt · score gap 1.0000</div>
      <h4><code>image_static_stylized_public_spaces_0033</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: stylized / indoor / public_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Prompt target</span>
        <p>Camera orbit right. A library hall with bookshelves, chandeliers, and reading tables.</p>
        <small>prompt source: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Good output: image_static_stylized_public_spaces_0033">
      <source src="../../../assets/metric_examples/d463b04eb239__wan22-ti2v-5b-image-static-plan-missing__image_static_stylized_public_spaces_0033.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">Good output · Score 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Poor output: image_static_stylized_public_spaces_0033">
      <source src="../../../assets/metric_examples/7fd273eb810c__hy_worldplay_ar_rl_image_test_balanced_5000_5s__image_static_stylized_public_spaces_0033.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">Poor output · Score 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
