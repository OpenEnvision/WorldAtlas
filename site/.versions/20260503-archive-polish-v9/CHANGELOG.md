# 20260503-archive-polish-v9

- Hide unloaded WebGL media instead of showing pale placeholder rectangles; textures still queue in near/center priority order.
- Bias texture loading toward close, central images so the first-person view fills in from the camera direction.
- Reduce edge-cropped foreground dominance and increase readable mid/far layers for smoother depth gradients.
- Tighten the ring-like vertical field and lower dynamic scatter to reduce top/bottom dead zones and clumping.
- Increase WASD response and camera follow speed while retaining inertial damping.
