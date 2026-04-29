# Style Consistency

`style_consistency` 用来衡量预测结果在配对锚点帧上保留参考结果整体风格统计的程度。
它使用 ImageNet 预训练的 VGG19 [[Simonyan and Zisserman, 2014](#reference)] 作为特征提取器，在多个卷积层上比较预测帧和参考帧的 Gram matrix。
如果两张图在颜色、纹理、局部笔触和粗粒度外观统计上相似，它们的 VGG 特征通道相关性也会更接近，最终 style consistency 分数会更高。

## 这个指标测什么

- 色彩、纹理分布和粗粒度外观统计的相似性。
- 参考条件下的风格保持。
- 预测结果是否维持与参考锚点相近的视觉风格、材质感和局部统计。

## 原理说明

VGG19 是一个深层卷积网络，原本用于大规模图像分类。
在这里，它不是用来判断图像类别，而是作为固定的视觉特征提取器。
浅层卷积特征通常更关注颜色、边缘和局部纹理；稍深一些的层会编码更大感受野下的重复结构和粗粒度外观模式。
因此，在多层 VGG19 特征上比较参考帧和预测帧，可以得到一个关于整体风格是否一致的代理信号。

Gram matrix 用来描述同一层特征通道之间的相关性。
它会弱化精确空间位置，更多保留“哪些纹理和颜色模式共同出现”这类统计信息。
这也是神经风格迁移中常用的风格表示方式：两张图即使局部内容位置不同，只要纹理、色彩和通道统计接近，它们的 Gram matrix 仍会相似。
因此，`style_consistency` 适合衡量生成结果是否在参考约束下保持了整体视觉风格。

## 具体算法

1. WorldArena 会把参考锚点帧和预测锚点帧一一配对。对视频，配对由当前 official 协议帧决定；对图片，就是单张图配单张图。
2. 每一对图像都会先做 VGG19 特征提取所需的 resize 和 ImageNet normalize。
3. 默认 `reference` 后端使用 `torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)`，并冻结全部参数。
4. 在 VGG19 features 的第 `0`、`2`、`5`、`7`、`10` 层提取特征激活。
5. 对每个提取层 `l`，计算参考和预测特征的 Gram matrix，然后计算层距离 `d_{i,l} = MSE(Gram(ref_{i,l}), Gram(pred_{i,l}))`。
6. 帧距离定义为 `d_i = mean_l d_{i,l}`。
7. 帧相似度定义为 `s_i = clamp(1 / (1 + d_i), 0, 1)`。
8. 原始指标值为 `raw = mean_i s_i`。由于当前输出本来就在 `[0, 1]`，benchmark 归一化实际上是恒等映射。

## 误差公式

对第 \(i\) 对参考帧和预测帧，记它们在 VGG19 第 \(l\) 个选定层的特征为 \(F^{ref}_{i,l}\) 和 \(F^{pred}_{i,l}\)。
若特征形状为 \(C_l \times H_l \times W_l\)，先将空间维度展平为 \(C_l \times H_lW_l\)，Gram matrix 定义为：

!!! info "Gram matrix"

    \[
    G(F_{i,l})
    =
    \frac{
        F_{i,l}F_{i,l}^{\top}
    }{
        C_l H_l W_l
    }
    \]

其中，\(G(F_{i,l})\) 表示该层不同特征通道之间的归一化相关性。
每一层的风格距离为：

!!! info "层级风格距离"

    \[
    d_{i,l}
    =
    \operatorname{MSE}
    \left(
        G(F^{ref}_{i,l}),
        G(F^{pred}_{i,l})
    \right)
    \]

单帧配对的平均风格距离和相似度为：

!!! info "帧级分数"

    \[
    d_i
    =
    \frac{1}{|\mathcal{L}|}
    \sum_{l\in\mathcal{L}} d_{i,l},
    \qquad
    s_i
    =
    \operatorname{clamp}
    \left(
        \frac{1}{1+d_i},
        0,
        1
    \right)
    \]

最终视频或样本级分数为全部配对锚点帧的平均值：

!!! info "最终分数"

    \[
    s_{\mathrm{style}}
    =
    \frac{1}{N}
    \sum_{i=1}^{N}
    s_i
    \]

其中，\(\mathcal{L}=\{0,2,5,7,10\}\) 是当前实现中使用的 VGG19 feature 层索引，\(N\) 是配对锚点帧数量。
分数越高，表示预测结果和参考结果在 VGG 风格统计上越接近。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 预测锚点帧 |
| 参考输入 | 配对的参考锚点帧 |
| 标注输入 | 无 |
| 原始输出 | `[0, 1]` 上的单个标量风格相似度 |
| 归一化输出 | 同一个 `[0, 1]` 标量 |
| 细节字段 | `frame_scores`、`native_frame_distances`、`native_layer_distances` |
| 失败形式 | 如果 VGG backend 不可用，则返回 `raw = null`、`normalized = null` 和错误字符串 |


## 参考文献 {#reference}

!!! quote "Very Deep Convolutional Networks for Large-Scale Image Recognition"

    Karen Simonyan and Andrew Zisserman.  
    arXiv:1409.1556, 2014.  
    [https://arxiv.org/pdf/1409.1556](https://arxiv.org/pdf/1409.1556)

??? note "BibTeX"

    ```bibtex
    @article{simonyan2014very,
        title = {Very deep convolutional networks for large-scale image recognition},
        author = {Simonyan, Karen and Zisserman, Andrew},
        journal = {arXiv preprint arXiv:1409.1556},
        year = {2014},
    }
    ```
