# Long-Sequence 诊断指标

WorldArena 内置了 `Long-Sequence` backend，用来从完整 prediction video 中提取长时运动和画质诊断信号。
这组指标不是只看单帧，也不是只看相邻两帧，而是把一段生成视频切成多个短 clip，再分析更长 horizon 上的运动连续性和视觉质量。

它主要回答的问题是：生成视频在更长 horizon 上是否还能保持运动连续性和画质。

这组指标是 prediction-only 诊断信号，不使用 benchmark reference rollout，也不需要标注。
因此它们更适合解释长视频退化模式，例如运动停滞、突兀时间伪影、画质随时间下降等。

## 覆盖的指标

| 指标名 | 包装后的 Long-Sequence 维度 | 在 WorldArena 中的角色 |
| --- | --- | --- |
| `long_sequence_motion_smoothness` | `motion_smoothness` | 基于视频插帧误差的运动平滑度 |
| `long_sequence_dynamic_degree` | `dynamic_degree` | 基于光流强度的运动是否充分发生 |
| `long_sequence_aesthetic_quality` | `aesthetic_quality` | 基于无参考美学模型的视觉审美质量 |
| `long_sequence_imaging_quality` | `imaging_quality` | 基于 MUSIQ 的成像清晰度和退化质量 |

## 原理说明

普通短程指标容易漏掉只在更长 rollout 后出现的问题。
Long-Sequence backend 会先把完整视频切成多个短 clip，然后把原生 long-video 例程用于运动和画质维度。

`motion_smoothness` 用 AMT-S 做中间帧插值，比较插值结果和真实中间帧的差异。
`dynamic_degree` 用 RAFT 估计光流强度，判断视频是否有足够运动。
`aesthetic_quality` 和 `imaging_quality` 分别使用质量模型对每个 clip 的帧进行无参考打分。

## 具体算法

1. WorldArena 读取完整 prediction video，并在临时 workspace 中准备 Long-Sequence 输入目录。
2. `worldarena.benchmark.long_video` 会把长视频切成 `split_clip/`，并生成每个维度需要的最小 manifest。
3. dispatcher 根据请求维度调用对应实现：
    - `motion_smoothness`：AMT-S 插帧后比较预测中间帧和真实中间帧。
    - `dynamic_degree`：RAFT 估计光流，并统计高运动 transition 是否达到阈值。
    - `aesthetic_quality`：无参考 aesthetic scorer 对 clip 帧打分。
    - `imaging_quality`：MUSIQ 对 clip 帧打分。
4. 每个 WorldArena 指标只读取自己对应的 wrapped dimension。
5. wrapper 会记录运行时元数据，包括实际 device、source root、clip 数量、per-video scores 和维度特定细节。
6. 最后每个指标按 `config/benchmark.yaml` 中的 `[0,1]` 范围独立归一化；这四个指标都是越高越好。

## 关键公式

对 `motion_smoothness`，AMT-S 会用偶数采样帧插值得到中间帧，并和真实中间帧比较平均绝对差：

!!! info "运动平滑度"

    \[
    s_{\mathrm{smooth}}
    =
    \operatorname{clamp}
    \left(
        \frac{255-\operatorname{mean}(|I^{real}_{mid}-I^{interp}_{mid}|)}{255},
        0,
        1
    \right)
    \]

对 `dynamic_degree`，RAFT 会估计相邻采样帧之间的光流，并取光流幅值最高的 5% 像素均值作为 transition motion score。
当超过运动阈值的 transition 数量达到计数阈值时，该 clip 记为有动态，输出 `1.0`；否则输出 `0.0`。

对 `aesthetic_quality` 和 `imaging_quality`，backend 分别对每个 clip 的帧计算无参考质量分数，再对 clip、video 和整批样本求平均。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整 prediction video |
| 参考输入 | 无 |
| 标注输入 | 无 |
| 原始输出 | 每个 wrapped Long-Sequence 指标各自产出一个 `[0,1]` 标量，越高越好 |
| 归一化输出 | 每个指标各自产出一个 `[0,1]` 标量，当前配置下与原始方向一致 |
| 细节字段 | `dimension`、`duration_seconds`、`source_root`、`runtime_device`、`clip_result_count`、`video_count`、`per_video_scores`、帧索引元数据，以及维度特定字段 |
| 失败形式 | 如果内置 backend、本地运行时依赖缺失，或者 `Long-Sequence` 没有返回目标维度的值，则返回 `raw = null`、`normalized = null` 和错误字符串 |

## 运行时说明

- 当前 wrapper 通过 `benchmark.metric_runtime.long_sequence` 读取旁路运行时配置。
- AMT / RAFT 源码固定在 `thirdparty/AMT` 和 `thirdparty/RAFT`。
- long-sequence checkpoint 现在统一从 `benchmark.metric_runtime.long_sequence` 解析到唯一 checkpoint 根目录：
  `ckpt/amt_model/amt-s.pth`、`ckpt/raft_model/models/raft-things.pth`、`ckpt/musiq/musiq_spaq_ckpt-358bb6af.pth`。
- 共享的 HF/Torch cache 固定在 `ckpt/huggingface`、`ckpt/transformers` 和 `ckpt/torch`。
- 原生路径已经不再依赖 `decord`、`moviepy` 和 `easydict`。但 quality 相关路径仍然依赖 `pyiqa`，并需要兼容的 `timm` 版本。
- 如果启用 `use_semantic_splitting=true`，运行环境里还需要安装 `scenedetect`。
- 常见运行时参数包括 `device`、`use_semantic_splitting`、`clip_duration_seconds`、`clip_fps`、`clip_length_config` 和 `imaging_quality_preprocessing_mode`。

## 集成说明

- 当前聚合状态：在 `image_static`、`image_dynamic` 和 `video_dynamic` 中以 `diagnostic` 形式产出。
- 这组指标对长时 horizon 分析很有价值，但当前不进入 official headline 向量。
- 解读时应和 `style_consistency`、`perceptual_quality`、`motion_magnitude`、`motion_smoothness` 等主指标一起看：Long-Sequence 分数高说明长时运动和画质统计更稳定，不等价于 prompt 完全正确，也不等价于物理运动真实。
