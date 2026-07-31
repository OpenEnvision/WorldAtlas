# WorldArena Model & Evaluation Checkpoint Inventory

> 交接文档：给其他 agent 使用。本文记录 WorldArena 当前模型配置、官方代码仓库、模型 checkpoint、评测 checkpoint 和下载来源。
>
> 路径均相对于仓库根目录 `/mnt/cpfsB/yangboxue/visual_generation/juanxi/WorldArena`。以 `config/models/*.yaml` 和实际 runner 代码为准；如果本文与代码不一致，应优先修正本文。

## 1. 使用规则

- 模型权重统一放在 `ckpt/` 下，不要把大文件提交到 git。
- 配置里的 `checkpoint_dir` 是模型加载路径；评测脚本中的路径是评测权重加载路径。
- 标记为 **exact** 的文件名是当前代码明确加载的文件名，不能只下载同名目录中的其他权重。
- 标记为 **aux** 的内容是模型运行所依赖的 VAE、T5/UMT5、深度估计、图像编码器或 tokenizer 等辅助权重。
- 评测前先运行：

  ```bash
  bash scripts/download_metric_checkpoints.sh
  bash scripts/model_batch_preflight.py --help
  ```

- 如果 checkpoint 不在默认 `ckpt/` 路径，优先通过配置或环境变量显式映射，不要修改公共代码中的默认路径。

## 2. 模型清单

| Model / config | Official repository | Main model checkpoint / required directory | Auxiliary checkpoints / notes |
|---|---|---|---|
| ABot-World 0.5B — `abot_world.yaml` | [amap-cvlab/ABot-World](https://github.com/amap-cvlab/ABot-World) | `ckpt/ABot-World-0-5B-LF/`；HF: `acvlab/ABot-World-0-5B-LF` | **exact:** `diffusion_pytorch_model.safetensors`, `Wan2.2_VAE.pth`, `taew2_2.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, `google/umt5-xxl/` |
| AlayaWorld 15B — `alayaworld.yaml`, `alayaworld_production_local.yaml` | [AlayaLab/AlayaWorld](https://github.com/AlayaLab/AlayaWorld) | `ckpt/AlayaWorld/merged_infer.safetensors` | `ckpt/gemma-3-12b-it-qat-q4_0-unquantized/`；DA3: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`；`alayaworld_production_local.yaml` 是本地 override |
| CamI2V — `cami2v.yaml` | [ZGCTroy/CamI2V](https://github.com/ZGCTroy/CamI2V) | `ckpt/CamI2V/512_cami2v_100k.pt`；HF: `MuteApo/CamI2V` | I2V inference checkpoint |
| Cosmos Predict2.5 2B — `cosmos_predict25_2b.yaml` | [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | **exact:** `ckpt/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt` | `ckpt/Cosmos-Reason1-7B/`；`ckpt/Cosmos-Predict2.5-2B/tokenizer.pth` |
| Cosmos Predict2.5 14B — `cosmos_predict25_14b.yaml` | [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | **exact:** `ckpt/Cosmos-Predict2.5-14B/base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt` | `ckpt/Cosmos-Reason1-7B/`；`ckpt/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth` |
| FlashWorld — `flashworld.yaml` | [imlixinyang/FlashWorld](https://github.com/imlixinyang/FlashWorld) | **exact:** `ckpt/imlixinyang--FlashWorld/model.ckpt` | Base: `ckpt/Wan2.2-TI2V-5B-Diffusers/`；HF: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| FlashWorld image-static — `flashworld_image_static.yaml` | [imlixinyang/FlashWorld](https://github.com/imlixinyang/FlashWorld) | **exact:** `ckpt/imlixinyang--FlashWorld/model.ckpt` | Same Wan2.2 TI2V base as FlashWorld |
| GEN3C — `gen3c.yaml` | [nv-tlabs/Gen3C](https://github.com/nv-tlabs/Gen3C) | `ckpt/nvidia--GEN3C-Cosmos-7B/model.pt`；HF: `nvidia/GEN3C-Cosmos-7B` | `nvidia/Cosmos-Tokenize1-CV8x8x8-720p/{encoder.jit,decoder.jit,mean_std.pt,image_mean_std.pt}`；`google-t5/t5-11b/`；`Ruicheng/moge-vitl/model.pt` |
| Hunyuan-GameCraft — `hunyuan_gamecraft.yaml` | [Tencent-Hunyuan/Hunyuan-GameCraft-1.0](https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0) | `ckpt/tencent--Hunyuan-GameCraft-1.0/mp_rank_00_model_states.pt` | Optional distill checkpoint: `mp_rank_00_model_states_distill.pt` |
| HunyuanWorld — `hunyuanworld.yaml` | [Tencent-Hunyuan/HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) | `ckpt/HunyuanWorld-1/` | `ckpt/ZIM/zim_vit_l_2092/` |
| HY-WorldPlay — `hy_worldplay.yaml` | [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | `ckpt/HY-WorldPlay/`；**exact action:** `ckpt/HY-WorldPlay/ar_distilled_action_model/model.safetensors` | Check the upstream repo for the complete base package |
| Infinite-World — `infinite_world.yaml` | [MeiGen-AI/Infinite-World](https://github.com/MeiGen-AI/Infinite-World) | `ckpt/MeiGen-AI--Infinite-World/` | Full package required by the upstream inference code |
| InSpatio-World — `inspatio_world.yaml` | [inspatio/inspatio-world](https://github.com/inspatio/inspatio-world) | **exact:** `ckpt/inspatio--world/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors` | `ckpt/DA3/`；`ckpt/taehv/taew2_1.pth` |
| LingBot-Video Dense — `lingbot_video_dense.yaml` | [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video) | `ckpt/lingbot-video-dense-1.3b/`；HF: `robbyant/lingbot-video-dense-1.3b` | Must contain `model_index.json`, `transformer/`, `text_encoder/`, `vae/` |
| LingBot-Video MoE — `lingbot_video_moe.yaml` | [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video) | `ckpt/lingbot-video-moe-30b-a3b/`；HF: `robbyant/lingbot-video-moe-30b-a3b` | Full MoE package required |
| LingBot-World base — `lingbot_world.yaml` | [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world) | `ckpt/lingbot-world-base-cam/`；HF: `robbyant/lingbot-world-base-cam` | High/low-noise models, Wan2.1 VAE and UMT5 are required |
| LingBot-World base-act preview — `lingbot_world_base_act_preview.yaml` | [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world) | `ckpt/lingbot-world-base-act-preview/`；HF: `robbyant/lingbot-world-base-act-preview` | Shares the base-cam dependencies |
| LingBot-World fast — `lingbot_world_fast.yaml` | [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world) | `ckpt/lingbot-world-fast/`；HF: `robbyant/lingbot-world-fast` | Also requires shared `ckpt/lingbot-world-base-cam/` |
| LucidDreamer — `luciddreamer.yaml` | [luciddreamer-cvlab/LucidDreamer](https://github.com/luciddreamer-cvlab/LucidDreamer) | `ckpt/stabilityai--stable-diffusion-2-inpainting/` | ZoeDepth dependencies are also needed |
| Lyra-1 — `lyra1.yaml` | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) | **exact:** `ckpt/Lyra/lyra_static.pt` | GEN3C Cosmos tokenizer, `google-t5/t5-11b/`, `Ruicheng/moge-vitl/model.pt` |
| Lyra-2 — `lyra2.yaml` | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) | `ckpt/Lyra-2.0/`；upstream HF: `nvidia/Lyra-2.0` | Expected subdirectories: `image_encoder/`, `lora/`, `model/`, `recon/`, `text_encoder/`, `vae/`；also needs `google/umt5-xxl/`, MoGe and DA3 |
| Matrix-Game 1 — `matrix_game_1.yaml` | [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | `ckpt/Matrix-Game-1.0/` | `vae/`, `dit/`, text encoder and tokenizer assets |
| Matrix-Game 2 — `matrix_game_2.yaml` | [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | `ckpt/Skywork--Matrix-Game-2.0/base_distilled_model/base_distill.safetensors` | Runner also loads `Wan2.1_VAE.pth` under the checkpoint root |
| Matrix-Game 3 — `matrix_game.yaml` | [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | `ckpt/Matrix-Game-3.0/` | Use the upstream directory layout |
| Matrix-Game 3.5 first-person — `matrix_game_35_first.yaml` | [Riemann-Dynamics/Matrix-Game-3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) | **exact:** `ckpt/Matrix-Game-3.5/Matrix-Game-3.5-Base/first-person.safetensors` | Also requires `ckpt/Matrix-Game-3.5/Wan2.2-TI2V-5B/`, DA3, UMT5; HF: `RiemannDynamics/Matrix-Game-3.5-Base` |
| Matrix-Game 3.5 third-person — `matrix_game_35_third.yaml` | [Riemann-Dynamics/Matrix-Game-3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) | **exact:** `ckpt/Matrix-Game-3.5/Matrix-Game-3.5-Base/third-person.safetensors` | Same Wan2.2 / DA3 / UMT5 dependencies as first-person |
| NeoVerse — `neoverse.yaml` | [IamCreateAI/NeoVerse](https://github.com/IamCreateAI/NeoVerse) | **exact:** `ckpt/Yuppie1204--NeoVerse/NeoVerse/reconstructor.ckpt` | Upstream HF reference: `Yuppie1204/NeoVerse` |
| Open Dreamer — `open_dreamer.yaml` | [next-state/open-dreamer](https://github.com/next-state/open-dreamer) | `ckpt/open-dreamer/<step>/` | Numbered Orbax checkpoint plus `tokenizer/` and `dynamics_ema/`；HF: `reactor-team/open-dreamer` |
| RealCam-I2V — `realcam_i2v.yaml` | [ZGCTroy/RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V) | **exact:** `ckpt/RealCam-I2V/RealCam-I2V_CogVideoX1.5-5B_50k.pt` | Base: `ckpt/CogVideoX1.5-5B-I2V/`；HF: `MuteApo/RealCam-I2V`；base HF: `zai-org/CogVideoX1.5-5B-I2V` |
| Stable Virtual Camera — `stable_virtual_camera.yaml` | [Stability-AI/stable-virtual-camera](https://github.com/Stability-AI/stable-virtual-camera) | `ckpt/stable-virtual-camera/modelv1.1.safetensors` | Use the upstream model config together with this file |
| WonderJourney — `wonderjourney.yaml` | [KovenYu/WonderJourney](https://github.com/KovenYu/WonderJourney) | SD2 inpainting: `ckpt/sd2-community--stable-diffusion-2-inpainting/` | **exact/default:** `ckpt/WonderJourney/dpt_beit_large_512.pt`, `ckpt/WonderJourney/sam_vit_h_4b8939.pth`, `ckpt/oneformer_coco_swin_large/` |
| WonderWorld — `wonderworld.yaml` | [KovenYu/WonderWorld](https://github.com/KovenYu/WonderWorld) | SD2 inpainting: `ckpt/sd2-community--stable-diffusion-2-inpainting/` | **exact:** `ckpt/WonderWorld/repvit_sam.pt`；`ckpt/prs-eth--marigold-v1-0/`；`ckpt/prs-eth--marigold-normals-v0-1/`；OneFormer cache |
| WorldCam — `worldcam.yaml` | [cvlab-kaist/WorldCam](https://github.com/cvlab-kaist/WorldCam) | **exact:** `ckpt/worldcam/finetuned_dit.safetensors` | Wan2.1 companion weights required |
| WorldFM — `worldfm.yaml` | [inspatio/worldfm](https://github.com/inspatio/worldfm) | `ckpt/worldfm/worldfm_2-step.pth` | `ckpt/worldfm/vae/` and `ckpt/moge-2-vitl-normal/` |
| YUME 1.5 — `yume.yaml` | [stdstu12/YUME](https://github.com/stdstu12/YUME) | `ckpt/Yume-5B-720P/` | Optional `ckpt/InternVL3-2B-Instruct/` is disabled in the current config |

### Models not included in the table

`config/models/template_video_api.yaml.example` and `config/models/kling_api.yaml.example` are examples, not local checkpoint-backed models. They are excluded from the model count.

## 3. Evaluation checkpoint inventory

### 3.1 Core metrics

| Checkpoint path | Used by | Source / official URL | Status / note |
|---|---|---|---|
| `ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth` | `depth_collision` | [Metric-Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small) | Current workspace contains this file |
| `ckpt/amt_model/amt-s.pth` | `long_sequence` | [lalala125/AMT](https://huggingface.co/lalala125/AMT) | Current workspace contains this file |
| `ckpt/raft_model/models/raft-things.pth` | `long_sequence` | [RAFT models.zip](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip) | Runner expects this exact path; an official copy also exists at `ckpt/official_metrics/models/raft-things.pth` |
| `ckpt/musiq/musiq_spaq_ckpt-358bb6af.pth` | MUSIQ / `long_sequence` | [IQA-PyTorch release](https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth) | Current workspace contains this file |
| `ckpt/Pulse-of-Motion/vc_common_10_60fps.ckpt` | Temporal consistency / FPS-related evaluation | [Visual Chronometer](https://huggingface.co/xiangbog/Visual_Chronometer) | Expected by the metric code; verify/download if absent |

### 3.2 Official metric/runtime weights

| Checkpoint path | Used by | Source / official URL |
|---|---|---|
| `ckpt/official_metrics/droid.pth` | DROID camera/reprojection | [DROID checkpoint](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh) |
| `ckpt/official_metrics/groundingdino_swint_ogc.pth` | GroundingDINO object grounding | [GroundingDINO release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| `ckpt/official_metrics/sam_vit_h_4b8939.pth` | SAM segmentation | [Segment Anything ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `ckpt/official_metrics/sam2.1_hiera_base_plus.pt` | SAM2 segmentation | [facebook/sam2.1-hiera-base-plus](https://huggingface.co/facebook/sam2.1-hiera-base-plus) |
| `ckpt/official_metrics/bert-base-uncased/` | BERT-based text/semantic metric | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| `ckpt/official_metrics/Tartan-C-T-TSKH-spring540x960-M.pth` | SEA-RAFT optical flow | [Yarimasune/SEA-RAFT](https://huggingface.co/Yarimasune/SEA-RAFT) |
| `ckpt/official_metrics/models/raft-things.pth` | RAFT optical flow | [RAFT models.zip](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip) |
| `ckpt/official_metrics/VFIMamba.pkl` | Motion smoothness / video feature metric | [MCG-NJU/VFIMamba](https://huggingface.co/MCG-NJU/VFIMamba) |
| `ckpt/official_metrics/sac+logos+ava1-l14-linearMSE.pth` | CLIP aesthetic score | [Improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth) |

### 3.3 Hugging Face and Torch cache dependencies

| Cache / checkpoint | Used by |
|---|---|
| `ckpt/huggingface/hub/openai/clip-vit-base-patch16` | Semantic similarity |
| `ckpt/huggingface/hub/openai/clip-vit-base-patch32` | Background consistency |
| `ckpt/huggingface/hub/facebook/dinov2-base` | Memory / DINO feature metric |
| `ckpt/huggingface/hub/depth-anything/Depth-Anything-V2-Small-hf` | Depth accuracy |
| `ckpt/huggingface/hub/depth-anything/DA3-GIANT-1.1` | Reconstruction / depth evaluation |
| `ckpt/huggingface/hub/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | MEt3R |
| `ckpt/mhamilton723--FeatUp/` | MEt3R feature backbone；HF: [mhamilton723/FeatUp](https://huggingface.co/mhamilton723/FeatUp) |
| `ckpt/torch/hub/checkpoints/pips2_weights.pth` | FVMD |
| `ckpt/torch/hub/checkpoints/vgg19-dcbb9e9d.pth` | Style / perceptual metric |
| `ckpt/torch/hub/pyiqa/brisque_svm_weights.pth` | BRISQUE |
| `ckpt/torch/hub/pyiqa/CLIP-IQA+_learned_prompts-603f3273.pth` | CLIP-IQA, if enabled |

Useful source links:

- [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)
- [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base)
- [Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT)
- [MASt3R metric model](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric)

### 3.4 Optional evaluation checkpoints

| Checkpoint / directory | Used by | Source / note |
|---|---|---|
| `ckpt/HPSv3/HPSv3.safetensors` and `ckpt/HPSv3_7B_local.yaml` | HPSv3 human preference score | [MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3) |
| `ckpt/Qwen2-VL-7B-Instruct/` | VLM-based evaluation | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| `ckpt/TransNetV2/transnetv2-pytorch-weights.pth` | Shot / scene boundary detection | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)；runner code under `thirdparty/TransNetV2/inference-pytorch/` |
| `ckpt/JEDi/vith16.pth.tar` | JEDi feature evaluation | [JEPA ViT-H weights](https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar) |
| `ckpt/JEDi/ssv2-probe.pth.tar` | JEDi probe | [SSv2 probe](https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar) |
| `thirdparty/mega-sam/` plus Depth Anything V1 ViT-L and RAFT | MegaSAM camera error | Camera-error path may use DROID instead; if MegaSAM is enabled, download its full external dependency tree |

The physics judge uses the OpenAI `gpt-4.1` API and does not require a local checkpoint. OpenCV epipolar, NumPy FFT, metadata, and pose/action metrics likewise do not require a learned checkpoint unless their runner explicitly selects one.

## 4. Checkpoint status observed in this workspace

The following files/directories were observed during the inventory pass. This is a snapshot, not a substitute for a preflight check:

- Populated main model roots: `ckpt/ABot-World-0-5B-LF/`, `ckpt/Matrix-Game-3.5/`, `ckpt/lingbot-world-base-cam/`.
- Populated metric/runtime files include `ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth`, `ckpt/amt_model/amt-s.pth`, `ckpt/musiq/musiq_spaq_ckpt-358bb6af.pth`, and several files under `ckpt/official_metrics/`.
- `ckpt/JEDi/`, `ckpt/HPSv3/`, `ckpt/Qwen2-VL-7B-Instruct/`, `ckpt/TransNetV2/`, and `ckpt/torch/` contain at least part of their expected assets.
- Verify these commonly missing or path-sensitive files before evaluation: `ckpt/raft_model/models/raft-things.pth`, `ckpt/Pulse-of-Motion/vc_common_10_60fps.ckpt`, and `ckpt/TransNetV2/transnetv2-pytorch-weights.pth`.
- Several model directories may exist but be empty or incomplete. Directory existence alone does not mean the model is runnable.

## 5. Agent handoff checklist

When adding or evaluating a model, the next agent should record:

1. The exact config filename under `config/models/`.
2. The official repository URL and, where applicable, the Hugging Face model URL.
3. The exact checkpoint path loaded by the runner.
4. All auxiliary weights required by the runner.
5. The evaluation metrics to run and their exact checkpoint paths.
6. The result of a local existence/size check, not just whether the parent directory exists.

Suggested quick checks:

```bash
rg -n "checkpoint_dir|ckpt|model_path|weights|pretrained" config/models scripts thirdparty
find ckpt -type f -printf '%s %p\n' | sort -n | tail -50
bash scripts/download_metric_checkpoints.sh --help
```

Do not silently rename checkpoint files. If a model requires a different local layout, update the model-specific config or document the mapping here with the reason.
