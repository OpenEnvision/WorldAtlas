# 20260504-refine-v29

- Expanded the camera-centered WebGL shell laterally and behind the viewer so turning keeps the archive field wrapped around the camera.
- Increased initial reveal readiness from 40 to 68 loaded WebGL textures to avoid exposing a half-loaded field.
- Smoothed WASD/wheel movement with softer acceleration and more responsive camera following.
- Rebalanced screen occupancy: larger cards remain readable, but oversized edge crops are constrained and weak distant overlaps fade harder.
- Extended near/side prefetch and fade hold so images transition instead of popping during view changes.
