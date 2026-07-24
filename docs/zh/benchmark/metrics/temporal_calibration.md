# Temporal Calibration

Temporal Calibration 度量生成视频是否维持**稳定的物理时钟**：当文件声明某个播放帧率时，画面中的运动是否对应相同的真实时间流逝？WorldArena 在 `real_time` 维度中以 output-only audit 形式调用被包装的 Pulse-of-Motion / Visual Chronometer 预测器 [[Gao et al., 2026](#reference-visual-chronometer)]。

这与 optical flow、motion magnitude 或 temporal smoothness 不同。一段 rollout 可以逐帧非常平滑，却在时间上仍然错误：坠落可能像被放慢的重力，快速周期性运动也可能显得不自然地迟缓。Visual Chronometer 将这类失败称为 **chronometric hallucination**——视觉连贯，但物理时间尺度模糊、不稳定或不可控。

预测器从重叠时间窗口估计 **Physical FPS**（`PhyFPS`），而不是单独相信容器元数据。WorldArena 将平均推断帧率与 nominal playback FPS $f_{\mathrm{meta}}$（OpenCV 读取的 container FPS，或 manifest 回退值）比较，并报告三个正式指标：`physical_fps_error`、`physical_fps_pct_error` 与 `physical_fps_intra_video_cv`。前两项刻画全局节奏偏差，第三项刻画单条视频内部的节奏漂移。

对每个 prediction，runtime 以默认参数 `clip_length = 30`、`stride = 4`、`resolution = 216` 提取重叠窗口，每个窗口得到一个片段估计 $\hat{f}_k$。平均推断物理 FPS 为

\[
\bar{f}_{\mathrm{phy}}
=
\frac{1}{K}\sum_{k=1}^{K} \hat{f}_k
\tag{1}
\]

绝对与相对标定误差分别为

\[
E_{\mathrm{abs}} = \left|\bar{f}_{\mathrm{phy}} - f_{\mathrm{meta}}\right|
\tag{2}
\]

\[
E_{\mathrm{pct}} = \frac{E_{\mathrm{abs}}}{f_{\mathrm{meta}}}
\tag{3}
\]

视频内部不稳定性由片段预测的变异系数表示：

\[
C_{\mathrm{intra}}
=
\frac{\operatorname{std}(\hat{f}_1,\ldots,\hat{f}_K)}{\operatorname{mean}(\hat{f}_1,\ldots,\hat{f}_K)}
\tag{4}
\]

WorldArena 将 `physical_fps_error`、`physical_fps_pct_error`、`physical_fps_intra_video_cv` 分别反向缩放到 $[0,1]$（`physical_fps_error` 使用 $[0,30]$，其余两项使用 $[0,1]$）。**归一化分数越高，表示时间标定越好。** 若 side runtime、checkpoint 或 FPS 元数据不可用，整组指标记为 `not_applicable`。

## 参考文献 {#reference-visual-chronometer}

!!! quote "The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"

    Xiangbo Gao, Mingyang Wu, Siyuan Yang, Jiongze Yu, Pardis Taghavi, Fangzhou Lin, and Zhengzhong Tu.  
    arXiv:2603.14375, 2026.
