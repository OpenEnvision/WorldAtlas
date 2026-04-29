# 运行时布局

WorldArena 现在只允许一个 checkpoint 根目录：

- `/bench-workspace/yesliu/hjd/model/WorldArena/ckpt`

所有 checkpoint 相关内容都统一落在这里：模型权重、benchmark 运行时权重、参考指标权重、Hugging Face cache、Transformers cache、Torch cache。`thirdparty/` 只保留源码 checkout，`cache/` 只保留临时运行产物。

## 硬约束

以下路径机制已经移除：

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend 各自维护的 checkpoint 搜索链
- 通过环境变量切换 checkpoint 根目录

当前固定缓存位置：

- `ckpt/huggingface`
- `ckpt/huggingface/hub`
- `ckpt/transformers`
- `ckpt/torch`

内置 adapter 和 backend 还会强制开启 HF/Transformers offline 模式。如果某个 checkpoint 缺失，代码现在会直接报清晰错误，不再回退到下载或其他目录。

## 目录契约

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

`cache/` 仍然只是可丢弃的工作目录，不再承担 checkpoint 职责。

## 模型映射

| 配置 | 源码 checkout | `ckpt/` 下的 checkpoint 路径 |
| --- | --- | --- |
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_world_base_act_preview.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-act-preview`，`generation.shared_checkpoint_dir=lingbot-world-base-cam` |
| `lingbot_world_fast.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam`，`generation.fast_checkpoint_dir=lingbot-world-fast` |
| `inspatio_world.yaml` | `thirdparty/inspatio-world` | `inspatio--world` |
| `infinite_world.yaml` | `thirdparty/Infinite-World` | `MeiGen-AI--Infinite-World` |
| `neoverse.yaml` | `thirdparty/NeoVerse` | `Yuppie1204--NeoVerse` |
| `lyra1.yaml` | `thirdparty/Lyra-1` | `Lyra`，以及 `nvidia--GEN3C-Cosmos-7B`、`nvidia--Cosmos-Tokenize1-CV8x8x8-720p`、`google-t5--t5-11b`、`Ruicheng--moge-vitl` |
| `lyra2.yaml` | `thirdparty/Lyra-2` | `Lyra-2.0`，以及 `depth-anything--DA3NESTED-GIANT-LARGE`、`google/umt5-xxl`、`Ruicheng--moge-vitl` |
| `gen3c.yaml` | `thirdparty/Lyra-1` | `nvidia--GEN3C-Cosmos-7B`，以及 `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`、`google-t5--t5-11b`、`Ruicheng--moge-vitl` |
| `flashworld.yaml`、`flashworld_image_static.yaml` | `thirdparty/FlashWorld` | `imlixinyang--FlashWorld` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam`，以及 `Wan-AI--Wan2.1-T2V-1.3B` |
| `hunyuanworld.yaml` | `thirdparty/HunyuanWorld-1.0` | `HunyuanWorld-1`，以及 `ZIM/zim_vit_l_2092` |
| `luciddreamer.yaml` | `thirdparty/LucidDreamer` | `stabilityai--stable-diffusion-2-inpainting`，以及 `huggingface/` 下的上游缓存条目 |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay`，`HY-WorldPlay/ar_distilled_action_model/model.safetensors` |
| `wonderjourney.yaml` | `thirdparty/WonderJourney` | `WonderJourney/dpt_beit_large_512.pt`，以及 `sd2-community--stable-diffusion-2-inpainting` 和 `huggingface/` 下的上游缓存条目 |
| `wonderworld.yaml` | `thirdparty/WonderWorld` | `WonderWorld/repvit_sam.pt`、`sd2-community--stable-diffusion-2-inpainting`、`prs-eth--marigold-v1-0`、`prs-eth--marigold-normals-v0-1`，以及 `huggingface/` 下的上游缓存条目 |
| `yume.yaml` | `thirdparty/YUME` | `stdstu123--Yume-5B-720P`，`InternVL3-2B-Instruct` |
| `worldfm.yaml` | `thirdparty/worldfm` | `worldfm`，以及 `ZIM/zim_vit_l_2092` |
| `matrix_game_1.yaml` | `thirdparty/Matrix-Game-1` | `Matrix-Game-1.0` |
| `matrix_game_2.yaml` | `thirdparty/Matrix-Game/Matrix-Game-2` | `Skywork--Matrix-Game-2.0` |
| `matrix_game.yaml` | `thirdparty/Matrix-Game-3.0` | `Matrix-Game-3.0` |
| `hunyuan_gamecraft.yaml` | `thirdparty/Hunyuan-GameCraft-1.0` | `tencent--Hunyuan-GameCraft-1.0` |

## Benchmark 运行时权重

| 运行时项 | `ckpt/` 下的规范定位 |
| --- | --- |
| CLIP backend | `openai/clip-vit-base-patch16 -> models--openai--clip-vit-base-patch16/snapshots/<revision>` |
| DINOv2 backend | `facebook/dinov2-base -> models--facebook--dinov2-base/snapshots/<revision>` |
| Depth Accuracy | `depth-anything/Depth-Anything-V2-Small-hf -> models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/<revision>` |
| MET3R FeatUp backbone | `mhamilton723--FeatUp` |
| Long-Sequence AMT | `amt_model/amt-s.pth` |
| Long-Sequence RAFT | `raft_model/models/raft-things.pth` |
| Long-Sequence MUSIQ | `musiq/musiq_spaq_ckpt-358bb6af.pth` |
| Temporal Calibration | `Pulse-of-Motion/vc_common_10_60fps.ckpt` |

内置指标运行时使用的固定源码 checkout：

- `thirdparty/met3r`
- `thirdparty/Pulse-of-Motion`
- `thirdparty/AMT`
- `thirdparty/RAFT`
- `thirdparty/Depth-Anything-3`

这些源码位置不再通过自定义 root 环境变量重定向。

## 路径解析规则

- project-relative 路径都相对于 WorldArena 仓库根目录解析。
- 所有 checkpoint 类路径都必须留在 `ckpt/` 下。
- 对于支持 checkpoint 树内子路径的 adapter，例如 `InSpatio-World`、`WorldFM`、`Matrix-Game-2`，相关子路径会在各自的 `checkpoint_dir` 内解析。
- checkpoint 缺失时直接抛 `FileNotFoundError`。
- WorldArena 不再保留“先试 A，再试 B，再试 C”的兼容层。

## 第三方源码策略

`thirdparty/` 只用于源码 checkout 或符号链接。

实践建议：

- 尽量让 clone 名称与内置配置默认值保持一致
- 在本地实验记录中固定上游 commit，不要把源码提交进这里
- 每个仓库和权重发布都分别接受其上游许可证
- 不要把 checkpoint 存到 `thirdparty/` 里

## 上游特殊说明

- `lingbot-world` 仍然支持 `generation.shared_checkpoint_dir` 和 `generation.fast_checkpoint_dir`，但它们现在只能是显式的 `ckpt/...` 路径；runner 会基于这些显式目录物化上游所需布局。
- `InSpatio-World` 会在 `ckpt/inspatio--world` 内保留上游子目录名，例如 `InSpatio-World-1.3B/`、`DA3/` 和可选的 `taehv/`。
- `HunyuanWorld` 仍然要求 `thirdparty/HunyuanWorld-1.0` 周围保留上游辅助源码布局，包括 `ZIM`、`Real-ESRGAN`、`MoGe` 这些源码 checkout。ZIM checkpoint 本身则在运行时从 `ckpt/ZIM/zim_vit_l_2092` 映射进去，不再存放在 `thirdparty/` 下。
- `WonderJourney` 和 `WonderWorld` 都遵循 WorldScore 的输出契约：benchmark prediction 是由生成帧写出的 `.mp4`；如果出现 zip，只作为生成帧和 3D 资产的 sidecar。
- `LucidDreamer`、`WonderJourney` 和 `WonderWorld` 的部分上游代码仍然会走 repo-id 型加载。WorldArena 现在只把这些缓存固定到 `ckpt/` 并离线运行，因此评测前需要先把相应 HF 资产预加载到 `ckpt/` 下的固定缓存树。
- `WorldCam`、`Lyra-1`、`Lyra-2`、`GEN3C` 除了主 `checkpoint_dir` 外，还需要额外的 companion checkpoint 一并放在 `ckpt/` 下。
