# 20260504-refine-v24

- Added soft-frustum rendering and visibility hysteresis so nearby samples do not disappear abruptly while turning.
- Expanded camera-centered side/back prefetch and shell persistence to keep the viewer surrounded by data.
- Increased texture cache/concurrency and screen occupancy granularity for smoother density with less flashing.
- Softened WASD/wheel acceleration to make first-person movement less abrupt.
- Preserved instant archive menu/navigation toggles with no open-close animation.
