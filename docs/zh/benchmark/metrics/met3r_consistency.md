# MEt3R 一致性

**met3r_consistency** 基于 MEt3R [[Asim et al., 2025](#reference)]，衡量 benchmark 协议锚点处**参考视角与预测视角之间的多视角三维一致性**。它关注两帧画面是否在几何上描述同一个场景——例如同一面墙、同一物体轮廓、同一空间深度关系是否能在两个视角下互相解释——而不是逐像素是否与 ground truth 对齐，也不是单纯问“图像好不好看”。

随着图像/视频扩散模型被用于新视角合成和相机轨迹生成，评测越来越需要一种**不依赖采样真值、也不要求已知相机位姿**的一致性指标。FID、FVD 只能衡量分布是否接近训练数据；TSED、SED 等 epipolar 指标又通常需要相对位姿，且容易对局部几何违规过度敏感，却漏掉明显的结构漂移。MEt3R 的设计目标，是在无位姿输入的前提下给出连续、可解释的三维一致性信号，并与图像质量指标正交。

对任意图像对 \(I_1, I_2\)，MEt3R 先用 DUSt3R 恢复像素对齐的稠密点图，并将两帧表达在同一相机坐标系下；随后提取 DINO 特征，经 FeatUp 上采样后投影到共享视角，在有效重叠区域比较特征余弦相似度 \(S(I_1, I_2)\)。在特征空间而非 RGB 空间比较，可减弱光照、曝光等视角相关变化的影响；相比光流 warp 后的 PSNR/SSIM，也不容易把模糊误判成“更一致”。MEt3R 分数定义为

\[
\mathrm{MEt3R}(I_1, I_2)
=
1 - \frac{1}{2}\big(S(I_1, I_2) + S(I_2, I_1)\big)
\tag{1}
\]

原始分数落在 \([0, 2]\)，**越低越好**。真实视频通常形成一个略高于零的下界，这来自 DUSt3R 对齐误差；而结构明显不一致的生成结果会得到显著更高的分数。

在 WorldArena 中，runtime 会在 benchmark 协议的 anchor 位置上，将 reference anchor frames 与 prediction anchor frames 逐对送入被包装的 MEt3R 模型；默认使用 `backbone = mast3r`、`distance = cosine`。原始得分为各 anchor pair 分数的均值，并在 \([0, 2]\) 上反向归一化到 \([0, 1]\)（归一化分数越高表示一致性越好）。若某对图像几乎没有有效重叠区域，会记为当前距离下的最差原始值，而不是被误判为满分。MEt3R 计算成功时，该指标在所有 formal suite 中为 **official**，并写入 `consistency_3d_4d` 聚合；它可补充几何监督，但不能替代 camera error、重投影误差等真值几何指标。

<!-- met3r-representative-cases:start -->
## MEt3R 代表性对比

以下代表性对比从 2026-07-17 正式 `image_static` MEt3R 快照中筛选，并经过 frame strip 人工复核。每个展示固定一个 benchmark case——相同的输入图像与 eval-plan prompt target——并比较两个 prediction 视频。MEt3R 归一化分数越高越好，raw distance 越低越好。

不同 adapter 的实际 conditioning 可能不同，因此这些是 metric 诊断案例，不是受控的模型排名对比。精确路径、哈希、overlap 值及 report 来源见[机器可读 MEt3R 清单](../../../assets/metric_examples/met3r_representative_manifest.csv)。

<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/5cd19d15663a__image_static_photorealistic_aquatic_landscape_0098.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0098" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 可见畸变对应更低的多视角一致性</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0098</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera pan right. An aquatic landscape with pebble beach, calm sea, and blue sky.</p>
      </div>
      <p class="metric-case-note">Output A 保持了海滩结构；Output B 出现大幅视角畸变和黑边，有效重叠下降，MEt3R raw distance 随之升高。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output A: ac3d-image-static，水岸 case 0098">
          <source src="../../../assets/metric_examples/d6290f43f723__ac3d-image-static__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output A</strong>
        <span>MEt3R：<strong>0.9730</strong> normalized · 0.0540 raw · 0.7044 overlap</span>
      </figcaption>
    </figure>
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output B: LucidDreamer，水岸 case 0098">
          <source src="../../../assets/metric_examples/20dc16268a49__luciddreamer-image-static-8gpu-20260613_151844__image_static_photorealistic_aquatic_landscape_0098.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output B</strong>
        <span>MEt3R：<strong>0.6341</strong> normalized · 0.7317 raw · 0.3180 overlap</span>
      </figcaption>
    </figure>
  </div>
</article>

<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/f6e5fd26f831__image_static_photorealistic_workspaces_0106.jpg" alt="输入图像: image_static_photorealistic_workspaces_0106" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 投影重叠为空时记最差值</div>
      <h4><code>image_static_photorealistic_workspaces_0106</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / indoor / workspaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera move left. A workspaces scene with bookshelf, indoor plants, and adjacent wall.</p>
      </div>
      <p class="metric-case-note">Output B 没有保留参考 workspace。MEt3R 找不到有效投影重叠，因此按文档中的 worst-value policy 记为 raw 2.0、normalized 0.0。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output A: ac3d-image-static，workspace case 0106">
          <source src="../../../assets/metric_examples/c0b5d552bebf__ac3d-image-static__image_static_photorealistic_workspaces_0106.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output A</strong>
        <span>MEt3R：<strong>0.9948</strong> normalized · 0.0104 raw · 0.9237 overlap</span>
      </figcaption>
    </figure>
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output B: CameraCtrl，workspace case 0106">
          <source src="../../../assets/metric_examples/5a90ac712030__cameractrl_image_static__image_static_photorealistic_workspaces_0106.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output B</strong>
        <span>MEt3R：<strong>0.0000</strong> normalized · 2.0000 raw · 0.0000 overlap</span>
      </figcaption>
    </figure>
  </div>
</article>

<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/c857427716ae__image_static_photorealistic_living_spaces_0140.jpg" alt="输入图像: image_static_photorealistic_living_spaces_0140" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 不同运动幅度下的 anchor 一致性接近</div>
      <h4><code>image_static_photorealistic_living_spaces_0140</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera push in. A living room with sofa, bookshelf, and books.</p>
      </div>
      <p class="metric-case-note">Output A 的运动较小；Output B 做了更大幅、平滑的相机运动并露出书架。在被评估的 MEt3R anchor 上，两者都保持了较高的特征与投影视角一致性，因此得分非常接近。</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output A: ac3d-image-static，living space case 0140">
          <source src="../../../assets/metric_examples/3a76d4fa805b__ac3d-image-static__image_static_photorealistic_living_spaces_0140.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output A</strong>
        <span>MEt3R：<strong>0.9957</strong> normalized · 0.0087 raw · 0.5406 overlap</span>
      </figcaption>
    </figure>
    <figure class="metric-output-card">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="Output B: LongCat-Video，living space case 0140">
          <source src="../../../assets/metric_examples/d43209bc4d20__longcat-video-image-static-8gpu-20260612_133950__image_static_photorealistic_living_spaces_0140.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="metric-output-caption">
        <strong>Output B</strong>
        <span>MEt3R：<strong>0.9972</strong> normalized · 0.0055 raw · 0.5455 overlap</span>
      </figcaption>
    </figure>
  </div>
</article>

</div>
<!-- met3r-representative-cases:end -->

<figure markdown="span">
  ![MEt3R 方法概览](../../../assets/img/met3r_method_overview.jpg)
  <figcaption>MEt3R 使用多视角生成图像之间的几何和表征一致性来度量生成结果的 multi-view consistency。</figcaption>
</figure>

## 参考文献 {#reference}

!!! quote "MEt3R: Measuring Multi-View Consistency in Generated Images"

    Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen.  
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

??? note "BibTeX"

    ```bibtex
    @inproceedings{asim25met3r,
        title = {MEt3R: Measuring Multi-View Consistency in Generated Images},
        author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
        booktitle = {IEEE/CVF Computer Vision and Pattern Recognition ({CVPR})},
        year = {2025},
    }
    ```
