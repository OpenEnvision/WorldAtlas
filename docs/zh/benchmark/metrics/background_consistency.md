# 背景一致性（Background Consistency）

背景一致性度量生成视频中**场景外观的时序稳定性**。相邻帧之间若背景布局、纹理或全局场景上下文发生大幅跳变，得分会降低。

对解码帧 $x_i$，CLIP ViT-B/32 提取 L2 归一化嵌入 $f_i$。指标对相邻帧对余弦相似度取平均：

\[
S_{\mathrm{bg}}
=
\frac{1}{N-1}\sum_{i=1}^{N-1}\cos(f_i, f_{i+1})
\tag{1}
\]

其中 $N$ 为评测帧数，$\cos(\cdot,\cdot)$ 为 CLIP 图像特征之间的余弦相似度。WorldArena 将 $S_{\mathrm{bg}}$ 截断至 $[0, 1]$；得分越高表示背景越稳定。该指标**仅需 prediction**，且至少需要两帧。

## 参考文献 {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
