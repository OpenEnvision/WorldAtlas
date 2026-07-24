# Upstream attribution

The adapters in this directory integrate the metric orchestration previously
published with the official WorldArena metric code. That code is available
under the MIT License; its notice is retained in `LICENSE`.

Heavy implementations are not copied here. They are loaded from the pinned
submodules under `thirdparty/`:

- DROID-SLAM
- GroundingDINO
- SAM2
- SEA-RAFT
- VFIMamba

WorldArena-specific path resolution, runtime isolation, and small-input
compatibility hooks remain in this directory so the upstream checkouts can stay
unmodified.
