# Scripts

Small helpers around the `worldarena` CLI. Most workflows use the CLI directly; see [`README.md`](../README.md).

| Script | Purpose |
|--------|---------|
| [`download_metric_checkpoints.sh`](download_metric_checkpoints.sh) | Download benchmark metric weights into `ckpt/` |
| [`setup_matrix_game_35.sh`](setup_matrix_game_35.sh) | Bootstrap Matrix-Game 3.5 submodule, conda env, and checkpoints |
| [`launch_model_batch_8gpu.sh`](launch_model_batch_8gpu.sh) | Generic 8-GPU checkpoint-resident production; derives GPU groups, starts idle groups progressively, and rejects per-sample checkpoint reloads |
| [`launch_abot_world_8gpu.sh`](launch_abot_world_8gpu.sh) | ABot-World persistent production |
| [`launch_alayaworld_8gpu.sh`](launch_alayaworld_8gpu.sh) | AlayaWorld persistent long-horizon production |
| [`queue_alayaworld_production.sh`](queue_alayaworld_production.sh) | Wait for a free GPU, validate one 45-chunk resident rollout, then gate the full 8-GPU AlayaWorld run |
| [`launch_lingbot_video_8gpu.sh`](launch_lingbot_video_8gpu.sh) | LingBot-Video Dense/MoE persistent production |
| [`launch_lingbot_world_8gpu.sh`](launch_lingbot_world_8gpu.sh) | LingBot-World base/fast/act persistent production |
| [`launch_gen3c_8gpu.sh`](launch_gen3c_8gpu.sh) | GEN3C persistent production |
| [`launch_camera_i2v_8gpu.sh`](launch_camera_i2v_8gpu.sh) | CamI2V / RealCam-I2V persistent production |
| [`launch_cosmos_predict25_8gpu.sh`](launch_cosmos_predict25_8gpu.sh) | Cosmos-Predict2.5 2B/14B persistent production |
| [`launch_flashworld_8gpu.sh`](launch_flashworld_8gpu.sh) | FlashWorld image/video persistent production |
| [`launch_hunyuan_gamecraft_8gpu.sh`](launch_hunyuan_gamecraft_8gpu.sh) | Hunyuan-GameCraft persistent production |
| [`launch_infinite_world_8gpu.sh`](launch_infinite_world_8gpu.sh) | InfiniteWorld persistent production |
| [`launch_luciddreamer_8gpu.sh`](launch_luciddreamer_8gpu.sh) | LucidDreamer persistent production |
| [`launch_lyra_8gpu.sh`](launch_lyra_8gpu.sh) | Lyra-1/2 persistent production |
| [`launch_stable_virtual_camera_8gpu.sh`](launch_stable_virtual_camera_8gpu.sh) | Stable Virtual Camera persistent production |
| [`launch_wonder_8gpu.sh`](launch_wonder_8gpu.sh) | WonderJourney/WonderWorld persistent production |
| [`launch_worldfm_8gpu.sh`](launch_worldfm_8gpu.sh) | WorldFM persistent production |
| [`launch_yume_8gpu.sh`](launch_yume_8gpu.sh) | Yume-1.5 persistent production |
| [`run_eval_shards.py`](run_eval_shards.py) | Multi-GPU sharded `worldarena evaluate` |
| [`build_video_static_vipe_manifest.py`](build_video_static_vipe_manifest.py) | Build a `video_static` manifest from VIPE camera annotations |

### Typical usage

```bash
# Matrix-Game 3.5 one-time setup
bash scripts/setup_matrix_game_35.sh

# CamI2V on 8 GPUs
CAMERA_I2V_MODEL=cami2v bash scripts/launch_camera_i2v_8gpu.sh

# Any adapter explicitly verified as load-once. Single-GPU models become eight
# resident workers; an nproc_per_node=8 model becomes one resident 8-GPU worker.
WORLDARENA_BATCH_MODEL_CONFIG=config/models/abot_world.yaml \
  bash scripts/launch_model_batch_8gpu.sh

# Model-specific shortcuts
bash scripts/launch_abot_world_8gpu.sh
LINGBOT_VIDEO_VARIANT=moe bash scripts/launch_lingbot_video_8gpu.sh
LINGBOT_WORLD_VARIANT=base bash scripts/launch_lingbot_world_8gpu.sh
bash scripts/launch_gen3c_8gpu.sh
COSMOS_PREDICT25_VARIANT=14b bash scripts/launch_cosmos_predict25_8gpu.sh
FLASHWORLD_VARIANT=image bash scripts/launch_flashworld_8gpu.sh
LYRA_VARIANT=2 bash scripts/launch_lyra_8gpu.sh
WONDER_VARIANT=world bash scripts/launch_wonder_8gpu.sh

# Queue AlayaWorld behind its full 60-second smoke gate when GPUs are busy.
bash scripts/queue_alayaworld_production.sh

# Sharded evaluation
python scripts/run_eval_shards.py \
  --config config/benchmark.yaml \
  --predictions-root artifacts/predictions/my-model \
  --output-dir artifacts/benchmark/reports/my-model-sharded \
  --suites image_static
```

Generation sharding is built into the CLI:

```bash
worldarena generate --num-shards 8 --shard-index 0 ...
```
