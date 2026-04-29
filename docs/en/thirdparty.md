# Runtime Setup

WorldArena now uses a single checkpoint root:

- `/bench-workspace/yesliu/hjd/model/WorldArena/ckpt`

Everything checkpoint-related resolves under that directory: model weights, benchmark weights, reference metric weights, Hugging Face cache, Transformers cache, and Torch cache. `thirdparty/` keeps source checkouts only. `cache/` is scratch runtime output only.

## Hard Rule

Removed path mechanisms:

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend-specific checkpoint search chains
- env-switchable checkpoint roots

Current fixed cache locations:

- `ckpt/huggingface`
- `ckpt/huggingface/hub`
- `ckpt/transformers`
- `ckpt/torch`

WorldArena also forces offline HF/Transformers runtime mode for bundled adapters and backends. If a required checkpoint is missing, the code now raises a clear error instead of falling back to downloads or alternate roots.

## Directory Contract

```text
WorldArena/
  ckpt/
  thirdparty/
    lingbot-world/
    inspatio-world/
    Infinite-World/
    NeoVerse/
    Lyra-1/
    Lyra-2/
    FlashWorld/
    WorldCam/
    HunyuanWorld-1.0/
    LucidDreamer/
    HY-WorldPlay/
    WonderJourney/
    WonderWorld/
    YUME/
    worldfm/
    Hunyuan-GameCraft-1.0/
    Matrix-Game-1/
    Matrix-Game/
    Matrix-Game-3.0/
    met3r/
    AMT/
    RAFT/
    Pulse-of-Motion/
    Depth-Anything-3/
  cache/
```

`cache/` remains disposable workspace output. It is no longer a checkpoint mechanism.

## Model Mapping

| Config | Source checkout | Checkpoint path(s) under `ckpt/` |
| --- | --- | --- |
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_world_base_act_preview.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-act-preview`, `generation.shared_checkpoint_dir=lingbot-world-base-cam` |
| `lingbot_world_fast.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam`, `generation.fast_checkpoint_dir=lingbot-world-fast` |
| `inspatio_world.yaml` | `thirdparty/inspatio-world` | `inspatio--world` |
| `infinite_world.yaml` | `thirdparty/Infinite-World` | `MeiGen-AI--Infinite-World` |
| `neoverse.yaml` | `thirdparty/NeoVerse` | `Yuppie1204--NeoVerse` |
| `lyra1.yaml` | `thirdparty/Lyra-1` | `Lyra` plus `nvidia--GEN3C-Cosmos-7B`, `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`, `google-t5--t5-11b`, `Ruicheng--moge-vitl` |
| `lyra2.yaml` | `thirdparty/Lyra-2` | `Lyra-2.0` plus `depth-anything--DA3NESTED-GIANT-LARGE`, `google/umt5-xxl`, `Ruicheng--moge-vitl` |
| `gen3c.yaml` | `thirdparty/Lyra-1` | `nvidia--GEN3C-Cosmos-7B` plus `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`, `google-t5--t5-11b`, `Ruicheng--moge-vitl` |
| `flashworld.yaml`, `flashworld_image_static.yaml` | `thirdparty/FlashWorld` | `imlixinyang--FlashWorld` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam` plus `Wan-AI--Wan2.1-T2V-1.3B` |
| `hunyuanworld.yaml` | `thirdparty/HunyuanWorld-1.0` | `HunyuanWorld-1` plus `ZIM/zim_vit_l_2092` |
| `luciddreamer.yaml` | `thirdparty/LucidDreamer` | `stabilityai--stable-diffusion-2-inpainting` and any required upstream HF cache entries under `huggingface/` |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay`, `HY-WorldPlay/ar_distilled_action_model/model.safetensors` |
| `wonderjourney.yaml` | `thirdparty/WonderJourney` | `WonderJourney/dpt_beit_large_512.pt` plus `sd2-community--stable-diffusion-2-inpainting` and any required upstream HF cache entries under `huggingface/` |
| `wonderworld.yaml` | `thirdparty/WonderWorld` | `WonderWorld/repvit_sam.pt`, `sd2-community--stable-diffusion-2-inpainting`, `prs-eth--marigold-v1-0`, `prs-eth--marigold-normals-v0-1`, and any required upstream HF cache entries under `huggingface/` |
| `yume.yaml` | `thirdparty/YUME` | `stdstu123--Yume-5B-720P`, `InternVL3-2B-Instruct` |
| `worldfm.yaml` | `thirdparty/worldfm` | `worldfm` plus `ZIM/zim_vit_l_2092` |
| `matrix_game_1.yaml` | `thirdparty/Matrix-Game-1` | `Matrix-Game-1.0` |
| `matrix_game_2.yaml` | `thirdparty/Matrix-Game/Matrix-Game-2` | `Skywork--Matrix-Game-2.0` |
| `matrix_game.yaml` | `thirdparty/Matrix-Game-3.0` | `Matrix-Game-3.0` |
| `hunyuan_gamecraft.yaml` | `thirdparty/Hunyuan-GameCraft-1.0` | `tencent--Hunyuan-GameCraft-1.0` |

## Benchmark Runtime Weights

| Runtime item | Canonical locator under `ckpt/` |
| --- | --- |
| CLIP backend | `openai/clip-vit-base-patch16 -> models--openai--clip-vit-base-patch16/snapshots/<revision>` |
| DINOv2 backend | `facebook/dinov2-base -> models--facebook--dinov2-base/snapshots/<revision>` |
| Depth Accuracy | `depth-anything/Depth-Anything-V2-Small-hf -> models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/<revision>` |
| MET3R FeatUp backbone | `mhamilton723--FeatUp` |
| Long-Sequence AMT | `amt_model/amt-s.pth` |
| Long-Sequence RAFT | `raft_model/models/raft-things.pth` |
| Long-Sequence MUSIQ | `musiq/musiq_spaq_ckpt-358bb6af.pth` |
| Temporal Calibration | `Pulse-of-Motion/vc_common_10_60fps.ckpt` |

Fixed source checkouts for bundled metric runtimes:

- `thirdparty/met3r`
- `thirdparty/Pulse-of-Motion`
- `thirdparty/AMT`
- `thirdparty/RAFT`
- `thirdparty/Depth-Anything-3`

These source locations are no longer redirected by custom root environment variables.

## Path Resolution Rules

- Project-relative paths resolve against the WorldArena repo root.
- Checkpoint-like paths must stay under `ckpt/`.
- Adapters that support subpaths under a checkpoint tree, such as `InSpatio-World`, `WorldFM`, and `Matrix-Game-2`, resolve those subpaths inside their configured `checkpoint_dir`.
- Missing checkpoints now raise `FileNotFoundError` directly.
- WorldArena does not keep compatibility layers that try multiple roots in sequence.

## Third-Party Repository Policy

`thirdparty/` is for source checkouts or symlinks only.

Practical rules:

- keep clone names aligned with bundled config defaults when possible
- pin upstream commits in local notes or experiment logs, not by committing source here
- accept upstream licenses separately for each repo and each weight release
- do not store checkpoints inside `thirdparty/`

## Upstream Notes

- `lingbot-world` still supports `generation.shared_checkpoint_dir` and `generation.fast_checkpoint_dir`, but both must now be explicit `ckpt/...` paths. The runner materializes the upstream expected layout from those explicit local directories.
- `InSpatio-World` keeps upstream subdirectory names such as `InSpatio-World-1.3B/`, `DA3/`, and optional `taehv/` inside `ckpt/inspatio--world`.
- `HunyuanWorld` still expects its auxiliary source tree layout around `thirdparty/HunyuanWorld-1.0`, including the `ZIM`, `Real-ESRGAN`, and `MoGe` source checkouts. The ZIM checkpoint itself is now staged from `ckpt/ZIM/zim_vit_l_2092` at runtime instead of living under `thirdparty/`.
- `WonderJourney` and `WonderWorld` follow the WorldScore output contract: the benchmark prediction is a rendered-frame `.mp4`; zip archives, when present, are sidecars for generated frames and 3D assets only.
- `LucidDreamer`, `WonderJourney`, and `WonderWorld` still contain upstream repo-id based loading paths. WorldArena now pins those caches to `ckpt/` and runs offline, so preload the required HF assets into the fixed cache tree under `ckpt/` before evaluation.
- `WorldCam`, `Lyra-1`, `Lyra-2`, and `GEN3C` require companion checkpoints under `ckpt/` beyond their primary `checkpoint_dir`.
