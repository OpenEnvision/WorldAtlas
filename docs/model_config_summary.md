# WorldArena Model Configuration Summary

This table is generated from `WorldArena/config/models/*.yaml`, excluding backup, example, debug, and smoke configs. `Model Type` is simplified to `video`, `3D`, or `4D`: generic I2V/T2V models are labeled `video`, geometry/scene/camera-view expansion models are labeled `3D`, and action/trajectory/interactive world models are labeled `4D`. `Ability` denotes the WorldArena benchmark suites supported by the config. `Video Length(s)` is computed as `Frame Number / FPS` when both values are explicit; otherwise the direct duration field or local runner default is used when available. Rows are ordered as `video`, then `3D`, then `4D`.

| Model Type | Model Name | Ability | Version | Resolution | Video Length(s) | FPS | Frame Number |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video | cogvideo-x1.5-5b-i2v | image-static, image-dynamic, video-static, video-dynamic | X1.5-5B-I2V | 768x1360 | 5.06 | 16 | 81 |
| video | cosmos-framework-i2v | image-static, image-dynamic, video-static, video-dynamic | Predict2.5-14B | 720 | 7.88 | 24 | 189 |
| video | cosmos-predict2.5-14b | image-static, image-dynamic, video-static, video-dynamic | 2.5-14B | 720p | 5.81 | 16 | 93 |
| video | cosmos-predict2.5-2b | image-static, image-dynamic, video-static, video-dynamic | 2.5-2B | 720p | 5.81 | 16 | 93 |
| video | cosmos3-nano | image-static, image-dynamic, video-static, video-dynamic | Cosmos3-Nano | 720 | 7.88 | 24 | 189 |
| video | echo-infinity-juanxi | image-static, image-dynamic | Juanxi config | 832x480 | 5.06 | 16 | 81 |
| video | helios-distilled-i2v | image-static, image-dynamic, video-static, video-dynamic | Distilled | 384x640 | 4.12 | 24 | 99 |
| video | hunyuan-video-i2v | image-static, image-dynamic | I2V | 720x1280 | 5.38 | 24 | 129 |
| video | kairos-sensenova-robot-4b-480p-distilled | image-static, image-dynamic, video-static, video-dynamic | 4B 480p Distilled | 480x640 | 3.38 | 24 | 81 |
| video | longcat-video-i2v | image-static, image-dynamic, video-static, video-dynamic | I2V | 480p | 6.20 | 15 | 93 |
| video | longlive-1.0 | image-static, image-dynamic, video-static, video-dynamic | 1.0 | 704x1280 | 5.33 | 24 | 128 |
| video | longlive-2.0 | image-static, image-dynamic, video-static, video-dynamic | 2.0 | 704x1280 | 5.33 | 24 | 128 |
| video | ltx-2.3-22b-distilled-i2v | image-static, image-dynamic, video-static, video-dynamic | 2.3-22B Distilled | 512x768 | 5 | 24 | 121 |
| video | lyra-1 | image-static, image-dynamic, video-static, video-dynamic | 1 | 704x1280 | 5.04 | 24 | 121 |
| video | lyra-1-juanxi | image-static, image-dynamic, video-static, video-dynamic | 1 Juanxi | 704x1280 | 5.04 | 24 | 121 |
| video | wan2.2-i2v-a14b | image-static, image-dynamic, video-static, video-dynamic | 2.2 I2V-A14B | 1280x720 | 5 | 16 | 81 |
| video | yume-1.5 | image-static, image-dynamic, video-static, video-dynamic | 1.5 | 704x1280 | source-dependent | 16 | source-dependent |
| 3D | ac3d-cogvideox-5b-i2v | image-static, image-dynamic | CogVideoX-5B I2V | 480x720 | 5 | 8 | 49 |
| 3D | astra | image-static, image-dynamic, video-static, video-dynamic | open-source demo | 480x832 | 1.20 | 20 | 24 |
| 3D | cameractrl | image-static, image-dynamic, video-static, video-dynamic | AnimateDiffV3/SD1.5 | 256x384 | not specified in config | not specified in config | 16 |
| 3D | flashworld | video-static | default | 480x704 | 1.60 | 15 | 24 |
| 3D | flashworld-image-static | image-static | image-static config | 480x704 | 1.60 | 15 | 24 |
| 3D | gen3c | image-static, video-static, video-dynamic | Cosmos-7B | 704x1280 | 5.04 | 24 | 121 |
| 3D | hunyuanworld | image-static, video-static | 1.0 | 3D asset output | not applicable | not video output | not video output |
| 3D | hunyuanworld-voyager-image-static | image-static | Voyager | 512x768 | 5.38 | 24 | 129 |
| 3D | luciddreamer | image-static, video-static | default | 512 | 5 | source-dependent | source-dependent |
| 3D | lyra-2 | image-static, video-static | 2.0 | 480x832 | 5.06 | 16 | 81 |
| 3D | motionctrl | image-static, image-dynamic | default | 256x256 | 1.60 | 10.0 | 16 |
| 3D | neoverse | video-static, video-dynamic | default | 336x560 | not specified in config | not specified in config | 81 |
| 3D | recammaster-wan2.1 | image-static | Wan2.1 | 480x832 | 2.70 | 30 | 81 |
| 3D | stable-virtual-camera | image-static, image-dynamic | 1.1 | 576x576 | 6.00 | 12 | 72 |
| 3D | warp-as-history | image-static, image-dynamic, video-static, video-dynamic | default | 384x640 | 5.06 | 16 | 81 |
| 3D | wonderjourney | image-static, video-static | default | 512 | 5.00 | 10 | 50 |
| 3D | wonderworld | image-static, video-static | default | 512 | 5.00 | 10 | 50 |
| 4D | dreamx-world-5b-cam | image-static, image-dynamic, video-static, video-dynamic | 5B-Cam | 704x1280 | 5.04 | 24 | 121 |
| 4D | hunyuan-gamecraft | image-static, image-dynamic, video-static, video-dynamic | 1.0 | 704x1216 | 1.38 | 24 | 33 |
| 4D | hunyuan-gamecraft-juanxi | image-static, image-dynamic | 1.0 Juanxi | 704x1216 | 1.38 | 24 | 33 |
| 4D | hy-worldplay | image-static, image-dynamic, video-static, video-dynamic | AR distilled | 480x832 | 5.21 | 24 | 125 |
| 4D | infinite-world | image-static, image-dynamic, video-static, video-dynamic | default | source-dependent | 5 | 30 | 81 |
| 4D | inspatio-world | video-static, video-dynamic | 1.3B | source-dependent | source-dependent | source-dependent | source-dependent |
| 4D | lingbot-world | image-static, image-dynamic, video-static, video-dynamic | base-cam | 480x832 | not specified in config | not specified in config | 81 |
| 4D | lingbot-world-base-act-preview | video-static, video-dynamic | base-act-preview | 832x480 | not specified in config | not specified in config | 81 |
| 4D | lingbot-world-fast | image-static, image-dynamic, video-static, video-dynamic | fast/base-cam | 832x480 | not specified in config | not specified in config | 81 |
| 4D | matrix-game | video-static, video-dynamic | 3.0 | 704x1280 | 5.76 | 17 | 98 |
| 4D | matrix-game-1 | video-static, video-dynamic | 1.0 | 1280x720 | 4.06 | 16 | 65 |
| 4D | matrix-game-2 | image-static, image-dynamic, video-static, video-dynamic | 2.0 | 352x640 | 12.50 | 12 | 150 |
| 4D | matrix-game-2-5s | image-static, image-dynamic, video-static, video-dynamic | 2.0 5s | 352x640 | 5.00 | 12 | 60 |
| 4D | matrix-game-3-5s | image-static, image-dynamic, video-static, video-dynamic | 3.0 5s | 704x1280 | 5.00 | 17 | 85 |
| 4D | matrix-game-3-5s-base | image-static, image-dynamic, video-static, video-dynamic | 3.0 5s base | 704x1280 | 5.00 | 17 | 85 |
| 4D | matrix-game-3-video-static-vipe | video-static | 3.0 VIPE | 704x1280 | source-dependent | 17 | source-dependent |
| 4D | sana-wm | image-static, image-dynamic, video-static, video-dynamic | WM | 704x1280 | 10.06 | 16 | 161 |
| 4D | worldcam | video-static | default | 480x832 | not applicable | match GT | match GT |
| 4D | worldfm | image-static, image-dynamic, video-static, video-dynamic | default | 512 | 2.70 | 30 | 81 |

Notes: `source-dependent` means the value follows the input sample or ground-truth clip at runtime. `not specified in config` means the value is not declared in the available WorldArena YAML/adapter files. For LongLive, the listed resolution/FPS are derived from the upstream Wan2.2-TI2V-5B inference config available in the local repo. For Echo-Infinity, `num_output_frames=21` latent frames are converted to 81 decoded video frames according to the YAML comment.
