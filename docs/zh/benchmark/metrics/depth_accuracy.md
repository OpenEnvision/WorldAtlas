# Depth Accuracy

Depth Accuracy 是配对指标，比较预测与参考锚帧的单目相对深度结构。两侧经同一 Depth Anything V2 模型推理，不使用 ground-truth 深度。

对第 \(i\) 对，设 \(D_i^{\mathrm{pred}}\)、\(D_i^{\mathrm{ref}}\) 为深度图。中位数比率对齐：

\[
\alpha_i = \frac{\mathrm{median}(D_i^{\mathrm{ref}})}{\max(\mathrm{median}(D_i^{\mathrm{pred}}), 10^{-6})},
\qquad
\tilde{D}_i^{\mathrm{pred}} = \alpha_i D_i^{\mathrm{pred}}
\tag{1}
\]

在有效像素 \(\Omega_i = \{(x,y) \mid D_i^{\mathrm{ref}}(x,y) > 10^{-3}\}\) 上，帧级绝对相对误差为

\[
e_i = \frac{1}{|\Omega_i|}\sum_{(x,y)\in\Omega_i} \frac{\left|\tilde{D}_i^{\mathrm{pred}}(x,y) - D_i^{\mathrm{ref}}(x,y)\right|}{\max(D_i^{\mathrm{ref}}(x,y), 10^{-6})}
\tag{2}
\]

原始得分 \(e_{\mathrm{depth}} = \frac{1}{N}\sum_i e_i\) 对 \(N\) 组对齐帧对取均值。误差越低，归一化得分越高。该指标评估粗粒度前后景排序是否保留，而非 metric 深度真值或相机轨迹精度。

<!-- metric-examples:start -->
## 暂无可核验的同 Case 对比

> 当前保留的正式报告只覆盖该指标的一条 prediction run，无法组成经过核验的同-case高/低分输出对。这里刻意不再展示不同 case 的极值。
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "Depth Anything V2"

    Yang et al., 2024.
    [https://arxiv.org/abs/2406.09414](https://arxiv.org/abs/2406.09414)
