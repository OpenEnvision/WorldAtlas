# Segment Continuity

Segment Continuity measures whether a generated video contains **detected scene cuts**. Unintended hard transitions break temporal coherence and indicate segment discontinuities.

TransNetV2 outputs a per-frame scene-boundary probability $p_i$ for decoded frame $x_i$. Frames with $p_i > \theta$ are flagged as cut candidates, with defaults $\theta = 0.5$. A cut is recorded at frame index $i$ when the binary mask rises from $0$ to $1$ at $i$, following TransNetV2 transition parsing.

To suppress spurious detections, cut candidates are filtered so that every retained scene spans at least $L$ frames (default $L = 10$). Let $n_{\mathrm{cut}}$ denote the number of retained cuts in one video. The per-video score is binary:

\[
S_{\mathrm{seg}}
=
\begin{cases}
1, & n_{\mathrm{cut}} = 0 \\
0, & n_{\mathrm{cut}} > 0
\end{cases}
\tag{1}
\]

Across $N$ evaluated videos, the model-level aggregate is

\[
\bar{S}_{\mathrm{seg}}
=
1 - \frac{n_{\mathrm{cut}}}{N}
\tag{2}
\]

where $n_{\mathrm{cut}}$ counts videos with at least one retained cut. WorldArena stores the per-sample binary value from Eq. (1) as the metric raw score. The metric is **prediction-only** and requires TransNetV2 inference code and weights.

## References {#reference}

!!! quote "TransNet V2 — Shot Boundary Detection"

    Sočik et al., ACM MM 2020.
    [https://arxiv.org/abs/2008.04838](https://arxiv.org/abs/2008.04838)
