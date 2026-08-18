# 时序闪烁（Temporal Flickering）

时序闪烁度量生成视频**相邻帧之间的像素稳定性**。帧间高频亮度或颜色跳变会产生可见闪烁；本指标采用 RGB 像素平均绝对误差（MAE）加以惩罚。

设解码帧序列为 $\{x_i\}_{i=1}^{N}$，相邻帧像素 MAE 为

\[
\mathrm{MAE}(x_i, x_{i+1})
=
\frac{1}{HWC}\sum_{h,w,c}\left|x_i(h,w,c)-x_{i+1}(h,w,c)\right|
\tag{1}
\]

其中 $H$、$W$、$C$ 分别为高、宽与通道数。WorldArena 在 uint8 RGB 取值区间 $[0,255]$ 上计算 MAE；若分辨率不一致，则将后一帧缩放至与前一帧相同后再作差。

时序闪烁得分对相邻帧 MAE 取平均：

\[
S_{\mathrm{flick}}
=
\frac{255 - \frac{1}{N-1}\sum_{i=1}^{N-1}\mathrm{MAE}(x_i, x_{i+1})}{255}
\tag{2}
\]

$S_{\mathrm{flick}}$ 越高表示闪烁越少。该指标**仅需 prediction**，且至少需要两帧。

<!-- temporal-flickering-representative-cases:start -->
## 已核验的同输入 Temporal Flickering 案例

本页只保留一个案例。两条视频的首帧都已通过与展示输入的几何匹配；分数仅用于解释当前 metric，不代表模型整体排名。

[机器可读审计清单](../../../assets/metric_examples/temporal_flickering_representative_manifest.csv).

<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/c857427716ae__image_static_photorealistic_living_spaces_0140.jpg" alt="输入图像: image_static_photorealistic_living_spaces_0140" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">核验案例 · 相邻像素 MAE 对正常相机运动敏感</div>
      <h4><code>image_static_photorealistic_living_spaces_0140</code></h4>
      <div class="metric-case-meta"><span>image_static</span><span>photorealistic / indoor / living_spaces</span></div>
      <div class="metric-case-prompt">
        <span>Benchmark prompt target</span>
        <p>Camera push in. A living room with sofa, bookshelf, and books.</p>
      </div>
      <p class="metric-case-note">Output A 的运动较小；Output B 做了更大幅、平滑的相机运动并露出书架。Output B 的低分反映更大的像素变化；由于当前实现没有做 motion compensation，仅凭该分数不能断言存在肉眼可见的高频闪烁。</p>
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
        <span>Temporal Flickering：<strong>0.9634</strong> · mean pairwise MAE 9.3268</span>
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
        <span>Temporal Flickering：<strong>0.7694</strong> · mean pairwise MAE 58.8134</span>
      </figcaption>
    </figure>
  </div>
</article>
</div>
<!-- temporal-flickering-representative-cases:end -->

## 参考文献 {#reference}

!!! quote "VBench — Temporal Flickering（附录 C.2.3）"

    相邻帧像素 MAE 经 $(255 - \overline{\mathrm{MAE}})/255$ 映射至 $[0,1]$ 稳定性得分。
