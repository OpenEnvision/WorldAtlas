# 20260504-refine-v15

- Increase mid/far WebGL readability by pushing fog farther and raising loaded media opacity in the distance gradient.
- Raise texture-load concurrency and cache size, with stronger anisotropy for clearer near images.
- Upgrade more near images to full resolution earlier while preserving loaded-only rendering.
- Wait for more loaded WebGL items before revealing the archive to reduce early sparse frames.
- Keep missing soundtrack state subtle: the control stays visually as Music instead of dominating the UI.
