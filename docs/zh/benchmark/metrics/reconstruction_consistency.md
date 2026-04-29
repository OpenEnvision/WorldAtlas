# 重建一致性

`reconstruction_consistency` 将 World-R1 的 `S_recon` 信号接入 WorldArena，作为诊断性的 3D/4D consistency 指标。它关注生成视频能否被重建成一个 3D Gaussian 表征，并沿输入视角路径重新渲染出一致的外观。

## 它测什么

- prediction video 的重建自一致性。
- Depth-Anything-3 Gaussian reconstruction 是否能保留预测视频中的可见外观。
- 它不是 prompt following 指标，不是 aesthetic quality，也不是 reference-video fidelity 指标。

## 当前算法

1. 指标需要 video prediction。
2. 默认读取 benchmark 已采样帧；`frame_source = full_video` 时读取完整预测视频。
3. Depth-Anything-3 从预测帧中估计相机姿态并重建 Gaussian splats，运行时设置 `infer_gs = true`。
4. 沿重建出的输入路径把 Gaussian 表征重新渲染成 GS video。
5. 用 LPIPS 比较 GS video 和原始 prediction frames。
6. 原始分数为：

    \[
    S_{\mathrm{recon}}
    =
    \mathrm{clip}(1 - \mathrm{mean}(\mathrm{LPIPS}), 0, 1)
    \]

7. 归一化在 `[0, 1]` 上恒等缩放；越高越好。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | prediction video 的采样帧或完整帧 |
| 参考输入 | 不使用 |
| 标注输入 | 不使用 |
| 后端 | `world_r1_lpips` |
| 运行时默认 | `repo_root = thirdparty/Depth-Anything-3/src`、`model_name = depth-anything/DA3-GIANT`、`scorer = lpips`、`lpips_net = alex`、`process_res = 504`、`gs_render_size = [512, 512]` |
| 原始输出 | `[0,1]` 内的 World-R1 `S_recon` 标量 |
| 归一化输出 | `[0,1]`，越高越好 |
| 细节字段 | `source_metric`、`frame_count`、`rendered_frame_count`、`render_size`、`process_res`、`trj_mode`、`model_name`、`source_root`、`package_name`、`lpips_mean`、`lpips_frame_distances`、`frame_scores` |

## 聚合状态

| 项目 | 状态 |
| --- | --- |
| Suite | `image_static`、`image_dynamic`、`video_static`、`video_dynamic` |
| 聚合 | 永远 `diagnostic` |
| 维度 | `consistency_3d_4d` |

运行时同时支持 WorldArena 自带的 `thirdparty/Depth-Anything-3/src/depth_anything_3` 布局，以及 World-R1 的 `reward_server.depth_anything_3` 包布局。
