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

## References {#reference}

!!! quote "HPSv3 — Human Preference Score"

    Vision-language reward model used to score image quality from human-preference supervision; WorldArena applies linear percentile normalization to per-video mean raw rewards.
