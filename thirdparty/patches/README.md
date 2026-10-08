# Runtime patches

These patches preserve source changes used by the local integrations. Each is
applied to the exact upstream commit recorded in `../sources.lock.json` and is
protected by a SHA-256 checksum. Run `python scripts/checkout_dependencies.py NAME`
from the project root; repeating the command recognizes an already applied patch.

| Source | Change |
| --- | --- |
| DROID-SLAM | Build CUDA kernels for Ampere and Hopper targets |
| FlashWorld | Allow the Wan VAE source to be selected through the environment |
| HY-WorldPlay | Respect group offload and release unused rollout KV caches |
| Matrix-Game-3.5 | Preserve the generated output duration |
| SolarWM-H3 | Stage and verify rollout latents before the decode phase |
| FVMD-frechet-video-motion-distance | Support current Matplotlib colormap APIs |

Upstream license and notice files remain in the fetched checkout. These patches
do not include internal regression tests or generated model outputs. Applying a
patch successfully is not a GPU inference validation.
