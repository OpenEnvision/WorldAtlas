# Reconstruction Consistency

`reconstruction_consistency` asks whether the generated video can be reconstructed into a 3D Gaussian representation that renders back consistently along the input view path.

## What It Measures

- reconstruction self-consistency of the prediction video
- whether a Depth-Anything-3 Gaussian reconstruction preserves the prediction's visible appearance
- not prompt following, not aesthetic quality, and not a reference-video fidelity metric

## Current Algorithm

1. The metric requires a video prediction.
2. It reads sampled benchmark frames by default; `frame_source = full_video` reads the full prediction.
3. Depth-Anything-3 reconstructs camera poses and Gaussian splats from the prediction frames with `infer_gs = true`.
4. The Gaussian representation is rendered back as a GS video along the reconstructed input path.
5. LPIPS compares the rendered GS video with the original prediction frames.
6. The raw score is:

    \[
    S_{\mathrm{recon}}
    =
    \mathrm{clip}(1 - \mathrm{mean}(\mathrm{LPIPS}), 0, 1)
    \]

7. Normalization is identity over `[0, 1]`; higher is better.

## Inputs And Outputs

| Item | Current implementation |
| --- | --- |
| Prediction input | sampled or full prediction video frames |
| Reference input | unused |
| Annotation input | unused |
| Backend | reconstruction LPIPS backend |
| Default runtime | `repo_root = thirdparty/Depth-Anything-3/src`, `model_name = depth-anything/DA3-GIANT`, `scorer = lpips`, `lpips_net = alex`, `process_res = 504`, `gs_render_size = [512, 512]` |
| Raw output | `S_recon` scalar in `[0, 1]` |
| Normalized output | `[0, 1]`, higher is better |
| Detail fields | `source_metric`, `frame_count`, `rendered_frame_count`, `render_size`, `process_res`, `trj_mode`, `model_name`, `source_root`, `package_name`, `lpips_mean`, `lpips_frame_distances`, `frame_scores` |

## Aggregation

| Item | Status |
| --- | --- |
| Suites | `image_static`, `image_dynamic`, `video_static`, `video_dynamic` |
| Aggregation | always `diagnostic` |
| Dimension | `consistency_3d_4d` |

The runtime supports both the bundled `thirdparty/Depth-Anything-3/src/depth_anything_3` layout and the external reconstruction-backend `reward_server.depth_anything_3` package layout.
