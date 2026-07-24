# Background Consistency

Background Consistency measures **temporal stability of scene appearance** in a generated video. Large shifts in background layout, texture, or global scene context between adjacent frames lower the score.

For each decoded frame $x_i$, CLIP ViT-B/32 extracts a normalized embedding $f_i$. The metric averages cosine similarity over consecutive frame pairs:

\[
S_{\mathrm{bg}}
=
\frac{1}{N-1}\sum_{i=1}^{N-1}\cos(f_i, f_{i+1})
\tag{1}
\]

where $N$ is the number of evaluated frames and $\cos(\cdot,\cdot)$ is cosine similarity between L2-normalized CLIP image features. WorldArena clamps $S_{\mathrm{bg}}$ to $[0, 1]$; higher values indicate more stable backgrounds. The metric is **prediction-only** and requires at least two frames.

## References {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
