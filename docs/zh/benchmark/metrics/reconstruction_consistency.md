# 重建一致性

**reconstruction_consistency** 是公开指标 **geometric_consistency** 与 **photometric_consistency** 共用的计算单元。Depth-Anything-3（DA3）从预测帧恢复深度与相机位姿，构建彩色点云，并评估这些点跨视角重投影的一致性。它只依赖 prediction 本身，不需要 reference video 或几何真值。

与旧版 World-R1 的 Gaussian splat + LPIPS `S_recon` 不同，也与 WorldLens 中对预渲染 reconstruction 视频做光流 warp + LPIPS 的 `novel_view_consistency` 不同，本指标直接在预测帧上运行 DA3（`infer_gs = false`），由深度图、估计位姿与 RGB 帧构建点云，再检验跨帧重投影是否与预测深度一致。因此它衡量的是**重建路径下的几何自洽性**，而不是 prompt following、感知质量或 reference-video fidelity。

对每一帧，backend 将其他帧观测到的点重投影到当前视角，并计算几何误差（重投影深度与预测深度图的 median relative depth error）与光度误差（z-buffer 渲染颜色相对原始帧的 PSNR）。headline 分数为几何一致性：

\[
\mathrm{geometric\_consistency}
=
\frac{1}{1 + \mathrm{mean}(\mathrm{valid\_geo\_errors})}
\tag{1}
\]

光度 PSNR 采用标准映射：

\[
\mathrm{photometric\_consistency} = 1 - 10^{-\mathrm{PSNR}/20}.
\tag{2}
\]

两个公开指标都落在 \([0, 1]\)，**越高越好**，并分别聚合。

默认路径对齐已发布参考实现 commit `47f0df55c1c15242eb546839750959bd9f36cf0d`：以 3 FPS 取帧，步长为 `round(source_fps / 3)`；DA3-GIANT-1.1 使用 `process_res = 504`；每帧最多用未设 seed 的 `torch.randperm` 抽取 5,000 个有效点。WorldArena 将 DA3 源码 revision 以及模型 config / weight 产物固定到该参考版本；直接传 RGB array 与经无损 PNG 中转在像素上等价。

为提高吞吐，WorldArena 通过一次顺序解码取得相同的目标帧，不再为每个目标帧反复 seek。启用该模式前，会对每个预测目录的代表视频做逐字节 parity 检查；目标帧索引和 RGB 像素必须完全相同，因此该优化不改变指标输入或得分逻辑。

采样后至少需要三帧。DA3 失败、帧数不足或点云构建失败时两个分数为 null。`geometric_consistency` 与 `photometric_consistency` 都作为 `consistency_3d_4d` 下的 official metric，并在每条视频上复用同一次 DA3 推理。WorldArena 保留单条视频的完整精度，到聚合序列化时才舍入。旧别名 `S_recon`、`score_reconstruction` 解析为 `geometric_consistency`；旧的 `reconstruction_consistency` 输出仍作为诊断兼容名保留。
