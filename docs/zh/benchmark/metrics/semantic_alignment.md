# Semantic Alignment

Semantic Alignment 衡量生成视频是否保留标注侧车（annotation sidecar）所描述的场景级内容，与 benchmark 提示 \(T\) 无关。它通过 caption 侧车衍生的参考文本评分，补充 Prompt Alignment 对显式生成指令的评估。

WorldArena 从 `sample.annotation_path` 下的 `caption.json` 构造参考文本 \(R\)，可选附加相机指令标签；若无标注文本，则回退至 `sample.prompt_current` 或 `sample.prompt_target`。对解码帧 \(f_i\)，CLIP ViT-B/16 给出 L2 归一化图像嵌入 \(\mathbf{e}_i\) 与文本嵌入 \(\mathbf{r} = \phi_T(R)\)。逐帧得分为

\[
\tilde{s}_i = \mathrm{clamp}\!\left(\frac{\cos(\mathbf{e}_i, \mathbf{r}) + 1}{2},\, 0,\, 1\right)
\tag{1}
\]

原始指标对所有评测帧取均值：

\[
S_{\mathrm{sem}} = \frac{1}{N}\sum_{i=1}^{N} \tilde{s}_i
\tag{2}
\]

\(S_{\mathrm{sem}}\) 越高，表示与标注参考描述的语义越接近。该指标不能替代指令跟随得分或物体级验证。

<!-- metric-examples:start -->
## 暂无可核验的同 Case 对比

> 旧示例来自 `artifacts/reports` 之外的 readiness/smoke 报告。它已从正式结果展示中移除，待保留报告快照中出现可核验的同-case对照后再补充。
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
