# 20260504-refine-v21

- Keeps the v20 360-degree ring-space layout.
- Adds low-priority peripheral prefetch for nearby side/back images so turning the camera does not reveal empty unloaded space.
- Raises texture concurrency and cache budget for smoother rotation while still loading on demand.
- Physics remains excluded from the visual exploration API.
