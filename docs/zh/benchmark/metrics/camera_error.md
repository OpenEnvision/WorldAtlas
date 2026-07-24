# Camera Error

Camera Error 量化从生成视频恢复的相机路径与标注 ground-truth 外参之间的**轨迹一致程度**，评估视点控制合规性，而非物体运动或指令跟随。

记 \(\{\mathbf{T}^{\mathrm{gt}}_t\}_{t=1}^{N}\) 与 \(\{\mathbf{T}^{\mathrm{pred}}_t\}_{t=1}^{N}\) 分别为标注与视觉里程计（MegaSAM 或 DROID-SLAM）得到的对齐 $4 \times 4$ camera-to-world 矩阵。从中分解旋转 \(\mathbf{R}_t\) 与平移 \(\mathbf{t}_t\)。平移经尺度对齐：

\[
c^\star = \arg\min_{c} \sum_{t=1}^{N} \left\|\mathbf{t}^{\mathrm{gt}}_t - c\,\mathbf{t}^{\mathrm{pred}}_t\right\|_2
\tag{1}
\]

凸优化不可用时回退至均值距离比。平均平移误差为

\[
E_t = \frac{1}{N}\sum_{t=1}^{N} \left\|c^\star \mathbf{t}^{\mathrm{pred}}_t - \mathbf{t}^{\mathrm{gt}}_t\right\|_2
\tag{2}
\]

平均旋转误差（度）由相对旋转 \(\mathbf{R}^{\mathrm{rel}}_t = \mathbf{R}^{\mathrm{pred}}_t (\mathbf{R}^{\mathrm{gt}}_t)^\top\) 给出：

\[
E_r = \frac{1}{N}\sum_{t=1}^{N} \arccos\!\left(\mathrm{clip}\!\left(\frac{\mathrm{tr}(\mathbf{R}^{\mathrm{rel}}_t) - 1}{2}, -1, 1\right)\right) \cdot \frac{180}{\pi}
\tag{3}
\]

WorldArena 将各分量按经验范围 \([0, 15°]\)（旋转）与 \([0, 0.5]\)（平移）归一化至 \([0,1]\)（越低越好），再取**几何均值**得到 \(S_{\mathrm{cam}} \in [0,1]\)。\(S_{\mathrm{cam}}\) 越高表示相机控制越接近标注。当存在位姿标注或协议对齐的合成相机路径时，该指标为 **official**。

<!-- metric-examples:start -->
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和完全一致的展示 prompt，然后比较模型生成结果。paired 展示必须满足 input image 相同、prompt 文本相同，并且一个模型高分、一个模型低分。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `camera_error`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/d6b1f731ad67__image_static_photorealistic_aquatic_landscape_0001.jpg" alt="输入图像: image_static_photorealistic_aquatic_landscape_0001" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 0.9189</div>
      <h4><code>image_static_photorealistic_aquatic_landscape_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_static</span><span>Scene: photorealistic / outdoor / aquatic_landscape</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera orbit left. A coastal landscape with ocean waves, distant hills, and sunset sky.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: image_static_photorealistic_aquatic_landscape_0001">
      <source src="../../../assets/metric_examples/1a70929b6c9a__camera_error_megasam_smoke_dreamx__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.9189</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: image_static_photorealistic_aquatic_landscape_0001">
      <source src="../../../assets/metric_examples/56c1b531c0e3__camera_error_corrected_smoke_dreamx__image_static_photorealistic_aquatic_landscape_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "DROID-SLAM — Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Teed & Deng, NeurIPS 2021.
    [https://arxiv.org/abs/2108.10869](https://arxiv.org/abs/2108.10869)
