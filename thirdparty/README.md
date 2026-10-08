# Third-party dependencies

WorldAtlas uses pinned upstream repositories under `thirdparty/`. The authoritative
source list is [`sources.lock.json`](sources.lock.json), including full commit IDs
and SHA-256 checksums for runtime patches.

```bash
python scripts/checkout_dependencies.py --list
python scripts/checkout_dependencies.py lingbot-world
python scripts/checkout_dependencies.py lingbot-world --check
```

The installer initializes nested submodules and applies recorded patches. It
refuses to reset existing checkouts at another revision. Use `--all` only if you
need every model and metric source.

`.gitmodules` registers the 16 dependencies stored as Git submodules. Additional
sources are installed explicitly by the same helper. Recursive cloning alone
does not prepare every model. The tables below describe their roles; the lock
file, including additional model variants, controls the actual revisions.

## Model sources

Only dependencies of retained leaderboard integrations are listed below.

| Source | Upstream | Configurations |
| --- | --- | --- |
| `ABot-World` | [amap-cvlab/ABot-World](https://github.com/amap-cvlab/ABot-World.git) | [abot_world/](../config/models/abot_world/) |
| `AlayaWorld` | [AlayaLab/AlayaWorld](https://github.com/AlayaLab/AlayaWorld.git) | [alayaworld/](../config/models/alayaworld/) |
| `AlayaWorld-v1.1` | [AlayaLab/AlayaWorld](https://github.com/AlayaLab/AlayaWorld.git) | [alayaworld/](../config/models/alayaworld/) |
| `Astronex-World` | [Astronex-Robotics/Astronex-World](https://github.com/Astronex-Robotics/Astronex-World.git) | [astronex_world/](../config/models/astronex_world/) |
| `DreamX-World` | [AMAP-ML/DreamX-World](https://github.com/AMAP-ML/DreamX-World.git) | [dreamx_world/](../config/models/dreamx_world/) |
| `Evoke` | [AlayaLab/Evoke](https://github.com/AlayaLab/Evoke.git) | [evoke/](../config/models/evoke/) |
| `FlashWorld` | [imlixinyang/FlashWorld](https://github.com/imlixinyang/FlashWorld.git) | [flashworld/](../config/models/flashworld/) |
| `GEN3C` | [nv-tlabs/Gen3C](https://github.com/nv-tlabs/Gen3C.git) | [gen3c/](../config/models/gen3c/) |
| `HY-WorldPlay` | [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay.git) | [hy_worldplay/](../config/models/hy_worldplay/) |
| `Helios` | [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios.git) | [helios/](../config/models/helios/) |
| `Infinite-World` | [MeiGen-AI/Infinite-World](https://github.com/MeiGen-AI/Infinite-World.git) | [infinite_world/](../config/models/infinite_world/) |
| `JoyAI-Echo` | [jd-opensource/JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo.git) | [echo_wm/](../config/models/echo_wm/) |
| `LTX-2.5` | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2.git) | [ltx25/](../config/models/ltx25/) |
| `LongCat-Video` | [meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video.git) | [longcat_video/](../config/models/longcat_video/) |
| `LucidDreamer` | [luciddreamer-cvlab/LucidDreamer](https://github.com/luciddreamer-cvlab/LucidDreamer.git) | [luciddreamer/](../config/models/luciddreamer/) |
| `Lyra-2` | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra.git) | [lyra2/](../config/models/lyra2/) |
| `Matrix-Game` | [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game.git) | [matrix_game/](../config/models/matrix_game/) |
| `Matrix-Game-3.5` | [Riemann-Dynamics/Matrix-Game-3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5.git) | [matrix_game/](../config/models/matrix_game/) |
| `RealCam-I2V` | [ZGCTroy/RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V.git) | [realcam_i2v/](../config/models/realcam_i2v/) |
| `Sana` | [NVlabs/Sana](https://github.com/NVlabs/Sana.git) | [sana_wm/](../config/models/sana_wm/) |
| `SolarWM` | [Junchao-cs/SolarWM](https://github.com/Junchao-cs/SolarWM.git) | [solarwm/](../config/models/solarwm/) |
| `SolarWM-H3` | [Junchao-cs/SolarWM](https://github.com/Junchao-cs/SolarWM.git) | [solarwm/](../config/models/solarwm/) |
| `Wan2.2` | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2.git) | [wan22/](../config/models/wan22/) |
| `Warp-as-History` | [yyfz/Warp-as-History](https://github.com/yyfz/Warp-as-History.git) | [warp_as_history/](../config/models/warp_as_history/) |
| `WorldCrafter` | [TencentARC/WorldCrafter](https://github.com/TencentARC/WorldCrafter.git) | [worldcrafter/](../config/models/worldcrafter/) |
| `WorldFoundry` | [OpenEnvision/WorldFoundry](https://github.com/OpenEnvision/WorldFoundry) | [cosmos3/](../config/models/cosmos3/), [minimax_h3/](../config/models/minimax_h3/) |
| `YUME` | [stdstu12/YUME](https://github.com/stdstu12/YUME.git) | [yume/](../config/models/yume/) |
| `lingbot-video` | [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video.git) | [lingbot_video/](../config/models/lingbot_video/) |
| `lingbot-world` | [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world.git) | [lingbot_world/](../config/models/lingbot_world/) |
| `lingbot-world-v2` | [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2.git) | [lingbot_world/](../config/models/lingbot_world/) |
| `ltx2.3` | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2.git) | [ltx23/](../config/models/ltx23/) |
| `stable-virtual-camera` | [Stability-AI/stable-virtual-camera](https://github.com/Stability-AI/stable-virtual-camera.git) | [stable_virtual_camera/](../config/models/stable_virtual_camera/) |
| `zing-world-model` | [seedleap/zing-world-model](https://github.com/seedleap/zing-world-model.git) | [zing_world/](../config/models/zing_world/) |

## Metric and shared runtime sources

These sources support evaluation or are shared by the retained adapters.

| Source | Upstream |
| --- | --- |
| `AMT` | [MCG-NKU/AMT](https://github.com/MCG-NKU/AMT.git) |
| `DROID-SLAM` | [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM.git) |
| `Depth-Anything-3` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3.git) |
| `FVMD-frechet-video-motion-distance` | [DSL-Lab/FVMD-frechet-video-motion-distance](https://github.com/DSL-Lab/FVMD-frechet-video-motion-distance) |
| `Grounded-Segment-Anything` | [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything.git) |
| `GroundingDINO` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO.git) |
| `JEDi` | [oooolga/JEDi](https://github.com/oooolga/JEDi) |
| `Prior-Depth-Anything` | [SpatialVision/Prior-Depth-Anything](https://github.com/SpatialVision/Prior-Depth-Anything.git) |
| `Pulse-of-Motion` | [taco-group/Pulse-of-Motion](https://github.com/taco-group/Pulse-of-Motion.git) |
| `RAFT` | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT.git) |
| `SEA-RAFT` | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT.git) |
| `UniDepth` | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth.git) |
| `VFIMamba` | [MCG-NJU/VFIMamba](https://github.com/MCG-NJU/VFIMamba.git) |
| `Video-Depth-Anything` | [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything.git) |
| `flash-attention` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention.git) |
| `met3r` | [mohammadasim98/met3r](https://github.com/mohammadasim98/met3r.git) |
| `sam2` | [haoyi-duan/sam2](https://github.com/haoyi-duan/sam2.git) |
| `tapnet` | [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) |
| `vipe` | [nv-tlabs/vipe](https://github.com/nv-tlabs/vipe.git) |

Memory-track ViPE sidecars run official NVIDIA ViPE from `thirdparty/vipe`.
UniDepth / Video-Depth-Anything / Prior-Depth-Anything load only from these
WorldAtlas checkouts for offline evaluation:

```bash
git submodule update --init \
  thirdparty/vipe \
  thirdparty/UniDepth \
  thirdparty/Video-Depth-Anything \
  thirdparty/Prior-Depth-Anything
```

Weights stay under `ckpt/` (HF snapshot `1d0d3c52`,
`Video-Depth-Anything-Small/video_depth_anything_vits.pth`,
`Prior-Depth-Anything/{depth_anything_v2_vitb.pth,prior_depth_anything_vitb.pth}`,
plus `ckpt/vipe/{droid.pth,geocalib-pinhole.tar}`).

## Per-model checkout

```bash
# Example: only LingBot-World
python scripts/checkout_dependencies.py lingbot-world

# Example: Matrix-Game 1/2/3 from one upstream repo
python scripts/checkout_dependencies.py Matrix-Game
```

Each model still needs its **own Python environment** and weights under `ckpt/`. Follow the upstream README in each `thirdparty/<Name>/` checkout. See [Getting Started](https://worldatlas.worldatlas-site.workers.dev/doc/getting-started/) for setup and validation scope.
