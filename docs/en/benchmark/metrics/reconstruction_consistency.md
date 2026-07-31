# Reconstruction Consistency

**reconstruction_consistency** is the shared compute unit behind the two public metrics **geometric_consistency** and **photometric_consistency**. Depth-Anything-3 (DA3) recovers depth and camera poses from prediction frames, builds a colored point cloud, and scores how consistently those points reproject across views. It requires only the prediction itself—no reference video or geometry ground truth.

Unlike the older World-R1 Gaussian-splat + LPIPS `S_recon` formulation, and unlike the WorldLens `novel_view_consistency` pipeline that scores pre-rendered reconstruction videos with optical-flow-warped LPIPS, this metric runs DA3 directly on prediction frames (`infer_gs = false`), builds a point cloud from depth maps, estimated poses, and RGB frames, and checks whether cross-frame reprojection agrees with the predicted depth. It therefore measures **geometric self-consistency under a reconstruction path**, not prompt following, perceptual quality, or reference-video fidelity.

For each frame, the backend reprojects points observed in other frames into the current view and computes geometric error (median relative depth mismatch between reprojected depth and the predicted depth map) and photometric error (PSNR between z-buffer rendered colors and the original frame). The headline score is geometric consistency:

\[
\mathrm{geometric\_consistency}
=
\frac{1}{1 + \mathrm{mean}(\mathrm{valid\_geo\_errors})}
\tag{1}
\]

Photometric PSNR uses the standard mapping:

\[
\mathrm{photometric\_consistency} = 1 - 10^{-\mathrm{PSNR}/20}.
\tag{2}
\]

Both public metrics lie in \([0, 1]\), with **higher being better**, and are aggregated independently.

The default path follows the published reference implementation at commit `47f0df55c1c15242eb546839750959bd9f36cf0d`: frames are sampled at 3 FPS with `step = round(source_fps / 3)`, DA3-GIANT-1.1 runs at `process_res = 504`, and at most 5,000 valid points are sampled per frame with an unseeded `torch.randperm` call. WorldArena pins the DA3 source revision and model config/weight artifacts to that reference. Passing RGB arrays directly is pixel-equivalent to staging frames through lossless PNG.

For throughput, WorldArena decodes the same requested indices in one sequential pass instead of repeatedly seeking the decoder. This changes no sampled index or RGB value: parity checks across representative videos from every prediction directory require byte-identical frames before the optimized mode is used.

At least three sampled frames are required. DA3 failure, insufficient frames, or point-cloud construction failure produces null scores. `geometric_consistency` and `photometric_consistency` are official metrics under `consistency_3d_4d`; both reuse one DA3 inference for each video. WorldArena retains full per-video precision until aggregate serialization. Legacy aliases `S_recon` and `score_reconstruction` resolve to `geometric_consistency`; the old `reconstruction_consistency` output remains available as a diagnostic compatibility name.

<!-- metric-examples:start -->
## No Verified Same-Case Pair Yet

> The retained formal snapshot contains only one completed model report for each reconstruction submetric (geometric and photometric consistency). A same-case high/low pair would therefore be fabricated, so no video comparison is shown.
<!-- metric-examples:end -->
