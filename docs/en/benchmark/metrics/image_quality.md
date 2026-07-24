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
