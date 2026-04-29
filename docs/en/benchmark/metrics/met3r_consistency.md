# MEt3R Consistency

`met3r_consistency` is a diagnostic multi-view consistency probe built on the wrapped MEt3R runtime.

<figure markdown="span">
  ![MEt3R method overview](../../../assets/img/met3r_method_overview.jpg)
  <figcaption>MEt3R measures multi-view consistency in generated images through geometric and representation consistency across views.</figcaption>
</figure>

## What It Measures {#what-it-measures}

- how similar paired reference and prediction anchor views look under the MEt3R scoring model
- multi-view or view-to-view consistency between reference and prediction anchors
- not a headline metric and not a replacement for geometry supervision

## Detailed Algorithm {#algorithm}

1. WorldArena pairs reference anchor frames with prediction anchor frames at the benchmark protocol positions.
2. It resolves the MEt3R source tree from the configured MEt3R checkout.
3. The runtime loads the configured model options, which default to:
    - `backbone = mast3r`
    - `distance = cosine`
    - `batch_size = 1`
4. Anchor-frame pairs are sent through the wrapped MEt3R model in batches.
5. The runtime returns one score per paired anchor view.
6. The raw metric is the mean of those per-pair scores.
7. Normalization follows the configured distance direction:
    - the default `cosine` distance uses the benchmark range `[0, 2]`
    - lower is better under the default config

## Inputs And Outputs {#inputs-outputs}

| Item | Current implementation |
| --- | --- |
| Prediction input | prediction anchor frames |
| Reference input | reference anchor frames |
| Annotation input | none |
| Raw output | one scalar mean MEt3R score |
| Normalized output | one scalar in `[0, 1]` after distance-aware scaling |
| Details payload | `frame_scores`, `pair_count`, `source_root`, `device`, `batch_size`, `img_size`, `use_norm`, `backbone`, `feature_backbone`, `feature_backbone_weights`, `upsampler`, `distance`, `freeze`, `native_direction` |
| Failure mode | returns `raw = null`, `normalized = null`, and an error string when the checkout, submodules, or side runtime are not ready |

## Integration Notes {#integration}

- Current aggregation: `diagnostic` in all formal suites.
- The wrapper caches results per prediction path and runtime settings, so repeated access to the same video is cheap after the first run.

## Reference {#reference}

!!! quote "MEt3R: Measuring Multi-View Consistency in Generated Images"

    Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen.  
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

??? note "BibTeX"

    ```bibtex
    @inproceedings{asim25met3r,
        title = {MEt3R: Measuring Multi-View Consistency in Generated Images},
        author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
        booktitle = {IEEE/CVF Computer Vision and Pattern Recognition ({CVPR})},
        year = {2025},
    }
    ```
