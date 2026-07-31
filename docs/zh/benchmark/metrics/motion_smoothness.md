# Motion Smoothness

Motion Smoothness 在帧插值先验下衡量生成视频的**时序可插值性**。VFIMamba 由相邻偶数帧重建被丢弃的奇数帧；重建误差越低表示运动越平滑连续，而非语义或幅度正确。

对窗口 \(k\)，\(\hat{I}_{2k+1} = \Phi(I_{2k}, I_{2k+2})\) 与真实奇数帧 \(I_{2k+1}\) 通过 MSE、SSIM、LPIPS 比较。原始分量对 \(K\) 个窗口取均值：

\[
E_{\mathrm{mse}} = \frac{1}{K}\sum_k \mathrm{MSE}(\hat{I}_{2k+1}, I_{2k+1}),\quad
S_{\mathrm{ssim}} = \frac{1}{K}\sum_k \mathrm{SSIM}(\hat{I}_{2k+1}, I_{2k+1}),\quad
E_{\mathrm{lpips}} = \frac{1}{K}\sum_k \mathrm{LPIPS}(\hat{I}_{2k+1}, I_{2k+1})
\tag{1}
\]

计算 MSE 前会先把解码得到的 `uint8` 帧转换为 `float32`，避免负像素差在减法和平方时按模 256 回绕。

各分量独立归一化后取算术均值。\(S_{\mathrm{smooth}}\) 越高表示生成运动越平滑。计算成功时为 **official**，无需参考视频。

当前配置中的 MSE 经验边界早于 `v2_float32_mse`。修复后的原始报告会带版本标记，但在用修正后的 benchmark 全量样本重新标定该边界之前，其归一化分数仍应视为临时值。

<!-- metric-examples:start -->
## 案例等待重生成

此前的对比使用 `v2_float32_mse` 之前的分数，现已主动撤下。人工复核过的视频资产与历史审计 manifest 继续保留；候选选择和展示分数必须基于修正后的报告重新生成。
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "VFIMamba: Video Frame Interpolation with State Space Models"

    Zhang et al., NeurIPS 2024.

!!! quote "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Zhang et al., CVPR 2018.
