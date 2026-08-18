# SSIM

**ssim** 用来衡量预测帧和参考帧之间的结构相似度。
SSIM 是 Structural Similarity Index Measure（结构相似性指标），它不是直接比较像素误差，而是比较两张图像在局部亮度、对比度和结构上的相似程度。
在 WorldArena 中，`ssim` 是一个 reference-based reconstruction 指标：预测帧越接近参考帧的局部结构和视觉统计，分数越高。

## 它测什么

- 相对参考的结构接近程度。
- 亮度、对比度和局部结构的一致性。
- 带一点感知偏置的 reconstruction 风格相似度。
- 它比 PSNR 更关注局部结构，但仍然不是语义、运动或物理正确性指标。

## 原理说明

PSNR 只关心像素误差大小，而 SSIM 试图更接近人眼对图像退化的感知方式。
它把两张图像的差异拆成三个方面：

- 亮度：局部平均值是否接近。
- 对比度：局部方差或标准差是否接近。
- 结构：去除亮度和对比度影响后，局部纹理结构是否相似。

WorldArena 的实现采用标准局部窗口形式。
它会对每个 RGB 通道分别用 Gaussian window 估计局部均值、方差和协方差，再得到一张 SSIM map。
最后对整张 SSIM map 求平均，并把 RGB 三个通道的分数继续平均成帧级分数。

## 当前算法

1. WorldArena 按对齐时间索引配对预测帧和参考帧。
2. 如果分辨率不一致，会先把预测帧 resize 到参考帧分辨率。
3. 将预测帧和参考帧转换为 `float32`。
4. 对每个 RGB 通道分别使用 `11x11`、`sigma = 1.5` 的 Gaussian window 估计局部统计量。
5. 用 \(C_1=(0.01\cdot255)^2\)、\(C_2=(0.03\cdot255)^2\) 稳定分母，计算 SSIM map。
6. 对每个通道的 SSIM map 求平均，得到通道分数。
7. 三个通道的 SSIM 平均成该帧分数，并钳制到 `[0,1]`。
8. 原始分数是所有帧 SSIM 的平均值。

## 误差公式

对第 \(i\) 对预测帧和参考帧，记单个颜色通道上的局部图像块为 \(x\) 和 \(y\)。
局部均值、方差和协方差分别为 \(\mu_x,\mu_y,\sigma_x^2,\sigma_y^2,\sigma_{xy}\)。
当前实现的单通道 SSIM map 为：

!!! info "局部 SSIM"

    \[
    \operatorname{SSIM}(x,y)
    =
    \frac{
        (2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)
    }{
        (\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)
    }
    \]

其中：

!!! info "稳定常数"

    \[
    C_1=(0.01\cdot255)^2,
    \qquad
    C_2=(0.03\cdot255)^2
    \]

对 RGB 图像，WorldArena 分别计算三个通道的平均 SSIM，再取通道平均：

!!! info "帧级 SSIM"

    \[
    s_i
    =
    \operatorname{clamp}
    \left(
        \frac{1}{3}
        \sum_{c\in\{R,G,B\}}
        \operatorname{mean}_{x,y}
        \left(
            \operatorname{SSIM}_{i,c}(x,y)
        \right),
        0,
        1
    \right)
    \]

最终视频或样本级分数是所有配对帧的平均值：

!!! info "最终分数"

    \[
    s_{\mathrm{SSIM}}
    =
    \frac{1}{N}
    \sum_{i=1}^{N}
    s_i
    \]

其中，\(N\) 是成功配对的帧数。
SSIM 越高，说明预测帧与参考帧在局部亮度、对比度和结构上越相似。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | 预测的采样帧 |
| 参考输入 | 参考的采样帧 |
| 标注输入 | 无 |
| 原始输出 | `[0, 1]` 内的单个平均 SSIM 标量 |
| 归一化输出 | 默认配置下仍是 `[0,1]` 有界标量，越高越好 |
| 细节字段 | `frame_scores`、帧索引元数据 |
| 失败方式 | 如果没有对齐帧可用，则返回空值和错误字符串 |

## 集成状态

| 项目 | 结论 |
| --- | --- |
| 当前聚合 | 仅 `diagnostic` |
| 论文可用性 | `仅作辅助` |

- 它的价值：比纯 PSNR 更关注结构。
- 它的限制：和 PSNR 一样，仍然会奖励复制参考，不能代替语义或控制正确性。
- 对轻微错位、非刚性变形、相机视角变化仍然敏感；高 SSIM 更像是“像参考”，不是“语义一定正确”。
- 建议用法：只作为 full-reference 辅助指标使用，并和 `psnr`、`style_consistency`、`perceptual_quality` 一起解读。

## 参考资料 {#reference}

!!! quote "结构相似性"

    结构相似性指标从亮度、对比度和结构三个方面衡量两张图像的相似程度。  
    [https://zh.wikipedia.org/wiki/結構相似性](https://zh.wikipedia.org/wiki/%E7%B5%90%E6%A7%8B%E7%9B%B8%E4%BC%BC%E6%80%A7)

<!-- metric-examples:start -->
## 已核验的同输入案例

每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。

Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.

<div class="metric-case-stack">
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/c2cb9cbb7479__image_static_photorealistic_passageways_0076__input.jpg" alt="输入图像: image_static_photorealistic_passageways_0076" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.7476</div>
      <h4><code>image_static_photorealistic_passageways_0076</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / passageways</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera move right. A library aisle with bookshelves, books, and reading table.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_passageways_0076">
          <source src="../../../assets/metric_examples/curated_20260801/5bc5be0ae204__image_static_photorealistic_passageways_0076__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9410</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_passageways_0076">
          <source src="../../../assets/metric_examples/curated_20260801/07118e8eb56d__image_static_photorealistic_passageways_0076__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.1934</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/5bb59c6bfb78__image_static_photorealistic_dining_spaces_0032__input.jpg" alt="输入图像: image_static_photorealistic_dining_spaces_0032" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.6786</div>
      <h4><code>image_static_photorealistic_dining_spaces_0032</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / indoor / dining_spaces</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera orbit left. An eclectic kitchen with martini sign, posters, and fruit bowl.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_dining_spaces_0032">
          <source src="../../../assets/metric_examples/curated_20260801/c0df8ae0b4fc__image_static_photorealistic_dining_spaces_0032__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.9218</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_dining_spaces_0032">
          <source src="../../../assets/metric_examples/curated_20260801/81c3f89d947c__image_static_photorealistic_dining_spaces_0032__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.2432</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/curated_20260801/e31f9eb4b75b__image_static_photorealistic_verdant_landscape_0001__input.jpg" alt="输入图像: image_static_photorealistic_verdant_landscape_0001" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">输入已核验 · 同一 benchmark case · gap 0.6761</div>
      <h4><code>image_static_photorealistic_verdant_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>套件: image_static</span><span>场景: photorealistic / outdoor / verdant_landscape</span></div>
      <div class="metric-case-prompt">
        <span>基准 prompt target</span>
        <p>Camera pull out. An indoor garden with flower beds, statues, and arched windows. Camera pan left. An indoor garden with side path, benches, and flower beds. Camera move left. An indoor garden with exit door, plants, and sunlight.</p>
        <small>prompt source: prompt_target</small>
      </div>
      <p class="metric-case-note">两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较高指标分: image_static_photorealistic_verdant_landscape_0001">
          <source src="../../../assets/metric_examples/curated_20260801/3b4d97b32e5b__image_static_photorealistic_verdant_landscape_0001__high.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">较高指标分 · 分数 0.7518</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较高指标分</strong><span>ssim</span></figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="较低指标分: image_static_photorealistic_verdant_landscape_0001">
          <source src="../../../assets/metric_examples/curated_20260801/258b319b48d6__image_static_photorealistic_verdant_landscape_0001__low.mp4" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">较低指标分 · 分数 0.0757</div>
      </div>
      <figcaption class="metric-output-caption"><strong>较低指标分</strong><span>ssim</span></figcaption>
    </figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->
