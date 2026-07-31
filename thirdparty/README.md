# Third-party dependencies

WorldArena vendors upstream model and metric code as **git submodules** under `thirdparty/`.
Paths in `config/models/*.yaml` use `repo_root: thirdparty/<Name>` relative to the repository root.

## Clone everything

```bash
git clone --recurse-submodules <worldarena-remote> WorldArena
cd WorldArena
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

`stable-virtual-camera` and some metric repos may pull nested submodules; keep `--recursive`.

## World models

| Submodule | Upstream | Model config(s) |
|-----------|----------|-----------------|
| `ABot-World` | [amap-cvlab/ABot-World](https://github.com/amap-cvlab/ABot-World) | `abot_world.yaml` |
| `AlayaWorld` | [AlayaLab/AlayaWorld](https://github.com/AlayaLab/AlayaWorld) | `alayaworld.yaml` |
| `CamI2V` | [ZGCTroy/CamI2V](https://github.com/ZGCTroy/CamI2V) | `cami2v.yaml` |
| `cosmos-predict2.5` | [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | `cosmos_predict25_*.yaml` |
| `FlashWorld` | [imlixinyang/FlashWorld](https://github.com/imlixinyang/FlashWorld) | `flashworld*.yaml` |
| `GEN3C` | [nv-tlabs/Gen3C](https://github.com/nv-tlabs/Gen3C) | `gen3c.yaml` |
| `HY-WorldPlay` | [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | `hy_worldplay.yaml` |
| `Hunyuan-GameCraft-1.0` | [Tencent-Hunyuan/Hunyuan-GameCraft-1.0](https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0) | `hunyuan_gamecraft.yaml` |
| `HunyuanWorld-1.0` | [Tencent-Hunyuan/HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) | `hunyuanworld.yaml` |
| `Infinite-World` | [MeiGen-AI/Infinite-World](https://github.com/MeiGen-AI/Infinite-World) | `infinite_world.yaml` |
| `inspatio-world` | [inspatio/inspatio-world](https://github.com/inspatio/inspatio-world) | `inspatio_world.yaml` |
| `lingbot-video` | [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video) | `lingbot_video_*.yaml` |
| `lingbot-world` | [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world) | `lingbot_world*.yaml` |
| `LucidDreamer` | [luciddreamer-cvlab/LucidDreamer](https://github.com/luciddreamer-cvlab/LucidDreamer) | `luciddreamer.yaml` |
| `Lyra-1`, `Lyra-2` | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) | `lyra1.yaml`, `lyra2.yaml` |
| `Matrix-Game` | [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | `matrix_game.yaml` → `Matrix-Game/Matrix-Game-3`; `matrix_game_1.yaml` → `Matrix-Game/Matrix-Game-1`; `matrix_game_2.yaml` → `Matrix-Game/Matrix-Game-2` |
| `Matrix-Game-3.5` | [Riemann-Dynamics/Matrix-Game-3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) | `matrix_game_35_*.yaml` |
| `NeoVerse` | [IamCreateAI/NeoVerse](https://github.com/IamCreateAI/NeoVerse) | `neoverse.yaml` |
| `open-dreamer` | [next-state/open-dreamer](https://github.com/next-state/open-dreamer) | `open_dreamer.yaml` |
| `RealCam-I2V` | [ZGCTroy/RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V) | `realcam_i2v.yaml` |
| `stable-virtual-camera` | [Stability-AI/stable-virtual-camera](https://github.com/Stability-AI/stable-virtual-camera) | `stable_virtual_camera.yaml` |
| `WonderJourney` | [KovenYu/WonderJourney](https://github.com/KovenYu/WonderJourney) | `wonderjourney.yaml` |
| `WonderWorld` | [KovenYu/WonderWorld](https://github.com/KovenYu/WonderWorld) | `wonderworld.yaml` |
| `WorldCam` | [cvlab-kaist/WorldCam](https://github.com/cvlab-kaist/WorldCam) | `worldcam.yaml` |
| `worldfm` | [inspatio/worldfm](https://github.com/inspatio/worldfm) | `worldfm.yaml` |
| `YUME` | [stdstu12/YUME](https://github.com/stdstu12/YUME) | `yume.yaml` |

Shared dependency used by several adapters:

| Submodule | Used by |
|-----------|---------|
| `Depth-Anything-3` | `inspatio_world.yaml`, `alayaworld.yaml`, Matrix-Game 3.5 setup |

## Benchmark / metric dependencies

| Submodule | Upstream | Role |
|-----------|----------|------|
| `AMT` | [MCG-NKU/AMT](https://github.com/MCG-NKU/AMT) | motion magnitude |
| `DROID-SLAM` | [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | camera metrics |
| `flash-attention` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | optional kernels |
| `Grounded-Segment-Anything` | [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | segmentation metrics |
| `GroundingDINO` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | detection metrics |
| `met3r` | [mohammadasim98/met3r](https://github.com/mohammadasim98/met3r) | 3D consistency |
| `Pulse-of-Motion` | [taco-group/Pulse-of-Motion](https://github.com/taco-group/Pulse-of-Motion) | motion quality |
| `RAFT` | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | optical flow |
| `sam2` | [haoyi-duan/sam2](https://github.com/haoyi-duan/sam2) | segmentation |
| `SEA-RAFT` | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) | motion magnitude |
| `VFIMamba` | [MCG-NJU/VFIMamba](https://github.com/MCG-NJU/VFIMamba) | motion smoothness |

## Per-model checkout

```bash
# Example: only LingBot-World
git submodule update --init thirdparty/lingbot-world

# Example: Matrix-Game 1/2/3 from one upstream repo
git submodule update --init thirdparty/Matrix-Game
```

Each model still needs its **own Python environment** and weights under `ckpt/`. Follow the upstream README in each `thirdparty/<Name>/` checkout. See also [`docs/en/thirdparty.md`](../docs/en/thirdparty.md).
