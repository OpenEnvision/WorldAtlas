# Long Sequence Aesthetic Quality

**long_sequence_aesthetic_quality** 衡量预测帧的**无参考美学吸引力均值**。每帧 $I_t$ 经冻结 CLIP ViT-L/14 编码，LAION-Aesthetics 线性头输出归一化至 $[0,1]$ 的逐帧分数。

\[
S_{\mathrm{aes}} = \frac{1}{T}\sum_{t=1}^{T} s_t^{\mathrm{aes}}
\tag{1}
\]

越高越好。该指标衡量长 rollout 质量，仅基于预测，由 `long_sequence` 后端（`aesthetic_quality` 维度）计算。共享运行时见 [Long-Sequence 指标](long_sequence.md)。

<!-- metric-examples:start -->
## 暂无可核验的同 Case 对比

> 当前保留候选中没有双侧通过输入核验的对照。原先包含六组案例的 gallery 已移除。
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "LAION-Aesthetics Predictor (CLIP-based linear aesthetic regressor)"

    Used via pyiqa `laion_aes`; CLIP ViT-L/14 features with a linear head trained on human aesthetic ratings.
