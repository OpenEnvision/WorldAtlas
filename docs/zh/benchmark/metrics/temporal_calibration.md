# Temporal Calibration

`temporal_calibration` 属于 WorldArena 顶层 `real_time` 维度。它关心的是生成视频是否拥有稳定的“物理时钟”：当视频文件声明自己以某个 FPS 播放时，画面里的真实运动节奏是否也对应同样的时间流速。

这和 optical flow、motion magnitude、motion smoothness 都不同。一段视频可以非常平滑，但仍然在时间上是错的：人倒向床面的速度可能像被放慢了重力，飞蛾振翅也可能被生成成不合理的慢动作。Visual Chronometer 论文把这类问题称为 **chronometric hallucination**：生成运动看起来连贯，却缺少明确、稳定、可控的物理时间尺度 [[Gao et al., 2026](#reference-visual-chronometer)]。

WorldArena 当前通过被包装的 `Pulse-of-Motion` / `Visual Chronometer` 预测器计算这组指标。预测器不是直接相信视频容器里的 nominal FPS，而是从视觉动态中估计 **Physical Frames Per Second**（`PhyFPS`）。

## 这个指标测什么

| 指标 | 原始值 | 含义 |
| --- | --- | --- |
| `physical_fps_error` | `abs(avg_phyfps - meta_fps)` | 平均物理节奏和声明 FPS 的绝对偏差 |
| `physical_fps_pct_error` | `physical_fps_error / meta_fps` | 相对节奏偏差，便于比较 12 FPS、24 FPS、30 FPS、60 FPS 等不同输出 |
| `physical_fps_intra_video_cv` | `std(segment_phyfps) / mean(segment_phyfps)` | 同一视频内部，不同重叠片段的物理 FPS 变异系数 |

三项指标合起来区分两类时间失败：

- **标定错误**：整段视频稳定地偏慢或偏快，和声明 FPS 不匹配。
- **节奏漂移**：视频不同片段暗示了不同的物理时钟，即使平均 FPS 看起来还可以。

这组指标测的是物理节奏合理性。它不衡量 prompt following、object control、camera accuracy、perceptual quality，也不判断运动语义是否正确。

## 原理说明

大多数视频文件都有一个元数据 FPS（`meta_fps`）。它决定播放器如何播放视频，但不保证画面中的运动真的对应这个真实时间尺度。互联网视频经常被慢放、加速、插帧、剪辑或重新编码；大型视频生成模型在这类数据上训练时，可能学会了视觉上合理的帧间过渡，却没有学到稳定的物理时钟。

Visual Chronometer 把缺失的量定义为 `PhyFPS`：由画面运动本身暗示的真实物理帧率。模型从短视频片段中预测 `log(PhyFPS)`，再通过指数变换还原出物理 FPS。它的训练数据来自高帧率视频的受控时间重采样，并显式模拟 sharp capture、motion blur 和 rolling shutter 等拍摄机制，因此模型学习的是运动线索，而不是简单依赖场景类别或元数据 [[Gao et al., 2026](#reference-visual-chronometer)]。

WorldArena 把这个思想用作 output-only audit：对每个生成视频，从帧内容中估计物理 FPS，再和 benchmark 实际用于播放的 FPS 比较。这样，Temporal Calibration 就成为一个 real-time / world-modeling 检查项：生成结果不能只是逐帧平滑，还应该维持位移和真实时间之间的合理关系。

## 具体算法

1. WorldArena 读取完整预测视频。
2. runtime 按当前默认参数提取重叠时间窗口：
    - `clip_length = 30`
    - `stride = 4`
    - `resolution = 216`
3. 每个窗口会被 resize 到 `216 x 216`，归一化到 `[-1, 1]`，并排成 `[C, T, H, W]`。
4. 被包装的 `Pulse-of-Motion` 模型对每个窗口预测 `log(PhyFPS)`。
5. runtime 对预测值取指数，得到每个窗口的 `predicted_phyfps`。
6. 对全部窗口预测取平均，得到 `avg_phyfps`。
7. `meta_fps` 的来源顺序是：
    - 如果 OpenCV 能读到预测视频的 container FPS，就用它
    - 否则退回到 benchmark manifest 里的 `sample.fps`
8. 计算三项指标：
    - `physical_fps_error = abs(avg_phyfps - meta_fps)`
    - `physical_fps_pct_error = physical_fps_error / meta_fps`
    - `physical_fps_intra_video_cv = std(segment_phyfps) / mean(segment_phyfps)`
9. WorldArena 对三项指标都使用有界反向缩放：
    - `physical_fps_error`：`[0, 30]`，越小越好
    - `physical_fps_pct_error`：`[0, 1]`，越小越好
    - `physical_fps_intra_video_cv`：`[0, 1]`，越小越好

## 公式

设生成视频被切成 \(K\) 个重叠窗口。预测器对第 \(k\) 个窗口估计出的物理 FPS 为 \(\hat{f}_k\)，视频容器或 manifest 给出的 nominal playback FPS 为 \(f_{\mathrm{meta}}\)。

!!! info "平均物理 FPS"

    \[
    \bar{f}_{\mathrm{phy}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    \hat{f}_k
    \]

平均标定误差为：

!!! info "绝对 FPS 误差"

    \[
    E_{\mathrm{abs}}
    =
    \left|
        \bar{f}_{\mathrm{phy}}
        -
        f_{\mathrm{meta}}
    \right|
    \]

相对标定误差为：

!!! info "相对 FPS 误差"

    \[
    E_{\mathrm{pct}}
    =
    \frac{
        E_{\mathrm{abs}}
    }{
        f_{\mathrm{meta}}
    }
    \]

视频内部时间不稳定性使用片段预测的变异系数表示：

!!! info "视频内部 PhyFPS 变异"

    \[
    C_{\mathrm{intra}}
    =
    \frac{
        \operatorname{std}(\hat{f}_1,\ldots,\hat{f}_K)
    }{
        \operatorname{mean}(\hat{f}_1,\ldots,\hat{f}_K)
    }
    \]

WorldArena 会分别保存这三个量。每项指标的归一化分数都是 `[0, 1]` 内的反向缩放值，所以归一化分数越高，表示时间标定越好。

## 如何读分数

| 分数模式 | 可能诊断 |
| --- | --- |
| 低 `physical_fps_error`，低 `physical_fps_intra_video_cv` | 节奏标定良好，视频有稳定物理时钟 |
| 高 `physical_fps_error`，低 `physical_fps_intra_video_cv` | 整段视频稳定偏慢或偏快，可能通过全局 retiming 改善 |
| 低 `physical_fps_error`，高 `physical_fps_intra_video_cv` | 平均节奏看似合理，但视频内部有加速或减速漂移 |
| 高 `physical_fps_error`，高 `physical_fps_intra_video_cv` | 严重 chronometric hallucination，平均节奏和局部稳定性都不可靠 |

这种读法和 Visual Chronometer 的评测设计一致：论文同时检查 metadata-vs-physical-FPS alignment 与 intra-video stability，并展示了基于 PhyFPS 的 retiming 能改善人类感知的时间自然度 [[Gao et al., 2026](#reference-visual-chronometer)]。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整预测视频 |
| 参考输入 | 无 |
| 标注输入 | 只有在 container FPS 缺失时，才退回使用 manifest 里的 FPS |
| 原始输出 | 每个 family 成员各自输出一个标量 |
| 归一化输出 | 反向缩放后的单个 `[0, 1]` 标量，越高越好 |
| 细节字段 | `avg_phyfps`、`segment_phyfps`、`segment_count`、`total_frames`、`meta_fps`、`meta_fps_source`、`device`、`clip_length`、`stride`、`resolution` |
| 失败形式 | 如果 side runtime、checkpoint 或 FPS 元数据不可用，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 当前集成是否足够严格

| 项目 | 结论 |
| --- | --- |
| 当前聚合状态 | 三个指标只要产出数值分数，当前都属于 `official` |
| 论文可用性结论 | `Supporting real-time / temporal-calibration metric` |

- 这组指标能暴露普通 flow、插帧 smoothness、perceptual quality 容易漏掉的一类失败：运动很顺，但物理时间不对。
- 它依赖外部预训练节奏估计器，也依赖可用的 FPS 元数据；因此更适合作为 calibrated audit signal，而不是绝对物理真值。
- 如果只重新编码预测视频并改变 container FPS，即便帧内容完全相同，`physical_fps_error` 和 `physical_fps_pct_error` 也可能变化。
- `physical_fps_intra_video_cv` 在完成片段预测后不再依赖 metadata，特别适合发现局部节奏漂移。
- 推荐用法：把这组指标和 motion smoothness、motion magnitude、semantic/control 指标一起报告。一个强 world-model 输出应该同时做到运动平滑、语义正确、物理时间合理。

## 参考文献 {#reference-visual-chronometer}

!!! quote "The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"

    Xiangbo Gao, Mingyang Wu, Siyuan Yang, Jiongze Yu, Pardis Taghavi, Fangzhou Lin, and Zhengzhong Tu.  
    arXiv:2603.14375, 2026.

??? note "BibTeX"

    ```bibtex
    @article{gao2026pulse,
        title = {The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics},
        author = {Gao, Xiangbo and Wu, Mingyang and Yang, Siyuan and Yu, Jiongze and Taghavi, Pardis and Lin, Fangzhou and Tu, Zhengzhong},
        journal = {arXiv preprint arXiv:2603.14375},
        year = {2026},
    }
    ```
