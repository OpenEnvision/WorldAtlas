# Motion Smoothness

`motion_smoothness` 用来衡量预测视频的生成运动是否平滑、连续，并且是否符合视频插帧模型学到的运动先验。
它的核心思想是：如果一段视频的运动是平滑的，那么只保留偶数帧后，标准视频帧插值模型应该能够较准确地重建被丢弃的奇数帧。
重建帧与原始奇数帧越接近，说明生成视频在时间上越可插值，运动突变和闪烁越少。

## 这个指标测什么

- 时间稳定性与可插值性。
- 相邻帧之间是否存在突兀跳变、闪烁或局部结构断裂。
- 生成运动是否符合视频帧插值模型的常见运动先验。

## 原理说明

视频帧插值模型通常学习的是“给定前后两帧，中间帧应该如何自然过渡”。
因此，它可以作为一种 motion prior：如果生成视频中的运动轨迹平滑，偶数帧之间应当足以推断中间的奇数帧；如果生成视频存在突变、抖动、闪烁或身份漂移，插值模型就难以重建原始奇数帧。

WorldArena 当前使用被包装的 `VFIMamba` [[Zhang et al., 2024](#reference-vfimamba)] backend 作为插帧模型。
评测时不需要参考视频，也不使用标注；它只检查 prediction video 自身的时间连续性。
具体做法是先丢弃奇数帧，得到一个低帧率版本，再用插帧模型从相邻偶数帧重建这些被丢弃的奇数帧，最后用 MSE、SSIM [[Wang et al., 2004](#reference-ssim)] 和 LPIPS [[Zhang et al., 2018](#reference-lpips)] 三个指标比较重建帧和原始奇数帧。

三个分量分别覆盖不同误差视角：

- `MSE` 衡量像素级重建误差，越低越好。
- `SSIM` [[Wang et al., 2004](#reference-ssim)] 衡量局部结构相似度，越高越好。
- `LPIPS` [[Zhang et al., 2018](#reference-lpips)] 衡量感知特征距离，越低越好。

三者归一化后取平均，可以同时考虑像素误差、结构保持和感知一致性。

## 具体算法

1. WorldArena 会把整段预测视频解码成 PNG 帧。
2. 被包装的第三方指标会把解码帧按索引拆成：
    - 偶数帧：`I_0, I_2, I_4, ...`
    - 奇数帧：`I_1, I_3, I_5, ...`
3. 丢弃奇数帧后，偶数帧构成低帧率视频。
4. 对每一对相邻偶数帧 `(I_{2k}, I_{2k+2})`，`VFIMamba` [[Zhang et al., 2024](#reference-vfimamba)] 会预测中间帧 `Î_{2k+1}`。
5. 预测出的中间帧再和真实奇数帧 `I_{2k+1}` 做三种比较：
    - `MSE`
    - `SSIM`
    - `LPIPS`
6. 原始指标值是对所有插值窗口求平均后的三分量结果：
    - `raw.mse = mean window MSE`
    - `raw.ssim = mean window SSIM`
    - `raw.lpips = mean window LPIPS`
7. WorldArena 对三个分量分别归一化：
    - `mse` 在 `[0.0, 82.4014]` 上，越小越好
    - `ssim` 在 `[0.9224, 1.0]` 上，越大越好
    - `lpips` 在 `[0.0, 0.0228]` 上，越小越好
8. 最终归一化分数是三个归一化分量的算术平均。

## 误差公式

给定生成视频帧序列 \(\{I_0,I_1,I_2,\ldots\}\)，对第 \(k\) 个插值窗口，视频帧插值模型 \(\Phi\) 根据相邻偶数帧预测中间帧：

!!! info "插值重建"

    \[
    \hat{I}_{2k+1}
    =
    \Phi(I_{2k}, I_{2k+2})
    \]

然后比较 \(\hat{I}_{2k+1}\) 和原始被丢弃的奇数帧 \(I_{2k+1}\)。
像素级误差为：

!!! info "MSE 分量"

    \[
    e^{\mathrm{mse}}_k
    =
    \operatorname{MSE}
    \left(
        \hat{I}_{2k+1},
        I_{2k+1}
    \right)
    \]

结构相似度分量为：

!!! info "SSIM 分量"

    \[
    s^{\mathrm{ssim}}_k
    =
    \operatorname{SSIM}
    \left(
        \hat{I}_{2k+1},
        I_{2k+1}
    \right)
    \]

感知距离分量为：

!!! info "LPIPS 分量"

    \[
    e^{\mathrm{lpips}}_k
    =
    \operatorname{LPIPS}
    \left(
        \hat{I}_{2k+1},
        I_{2k+1}
    \right)
    \]

对所有插值窗口求平均，得到原始三分量：

!!! info "原始三分量"

    \[
    E_{\mathrm{mse}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    e^{\mathrm{mse}}_k,
    \qquad
    S_{\mathrm{ssim}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    s^{\mathrm{ssim}}_k,
    \qquad
    E_{\mathrm{lpips}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}
    e^{\mathrm{lpips}}_k
    \]

WorldArena 分别把三个分量归一化到 `[0,1]`，其中 MSE 和 LPIPS 是反向缩放，SSIM 是正向缩放。
最终分数为：

!!! info "最终分数"

    \[
    s_{\mathrm{smooth}}
    =
    \frac{
        \tilde{s}_{\mathrm{mse}}
        +
        \tilde{s}_{\mathrm{ssim}}
        +
        \tilde{s}_{\mathrm{lpips}}
    }{3}
    \]

分数越高，表示插值模型越容易从相邻偶数帧恢复被丢弃的奇数帧，生成运动越平滑。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 完整解码后的预测视频 |
| 参考输入 | 无 |
| 标注输入 | 无 |
| 原始输出 | `{"mse": float, "ssim": float, "lpips": float}` |
| 归一化输出 | 经分量归一化和算术平均后的单个 `[0, 1]` 标量 |
| 细节字段 | `frame_count`、`native_components`、`frame_indices`、`sampled_frame_indices` |
| 失败形式 | 如果插值 runtime 失败，则返回 `raw = null`、`normalized = null` 和错误字符串 |


## 参考文献 {#reference-vfimamba}

!!! quote "VFIMamba: Video Frame Interpolation with State Space Models"

    Guozhen Zhang, Chunxu Liu, Yutao Cui, Xiaotong Zhao, Kai Ma, and Limin Wang.  
    Advances in Neural Information Processing Systems, 37:107225-107248, 2024.

??? note "BibTeX"

    ```bibtex
    @article{zhang2024vfimamba,
        title = {Vfimamba: Video frame interpolation with state space models},
        author = {Zhang, Guozhen and Liu, Chunxu and Cui, Yutao and Zhao, Xiaotong and Ma, Kai and Wang, Limin},
        journal = {Advances in Neural Information Processing Systems},
        volume = {37},
        pages = {107225--107248},
        year = {2024},
    }
    ```

### SSIM {#reference-ssim}

!!! quote "Image Quality Assessment: From Error Visibility to Structural Similarity"

    Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli.  
    IEEE Transactions on Image Processing, 13(4):600-612, 2004.

??? note "BibTeX"

    ```bibtex
    @article{wang2004image,
        title = {Image quality assessment: from error visibility to structural similarity},
        author = {Wang, Zhou and Bovik, Alan C and Sheikh, Hamid R and Simoncelli, Eero P},
        journal = {IEEE transactions on image processing},
        volume = {13},
        number = {4},
        pages = {600--612},
        year = {2004},
        publisher = {IEEE},
    }
    ```

### LPIPS {#reference-lpips}

!!! quote "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"

    Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.  
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

??? note "BibTeX"

    ```bibtex
    @inproceedings{zhang2018unreasonable,
        title = {The unreasonable effectiveness of deep features as a perceptual metric},
        author = {Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages = {586--595},
        year = {2018},
    }
    ```
