# Motion Accuracy

Motion Accuracy 衡量受监督动态区域内**前景运动显著性**。SAM2 将初始分割 mask 传播至整段预测；SEA-RAFT 估计相邻帧光流。该指标检验 mask 内物体区域是否比背景呈现更强运动——**不**将实际运动与目标轨迹、方向或速度曲线做严格对比。

对相邻帧对 \((I_t, I_{t+1})\)，设 \(\mathbf{u}_t\) 为光流场、\(M_t\) 为传播（膨胀后）物体 mask。定义幅值 \(m_t = \|\mathbf{u}_t\|_2\) 及

\[
\Delta_t = \max_{p \in M_t} m_t(p) - \max_{p \notin M_t} m_t(p)
\tag{1}
\]

原始得分为所有帧对的平均显著性：

\[
S_{\mathrm{mot}}^{\mathrm{raw}} = \frac{1}{N-1}\sum_{t=1}^{N-1} \Delta_t
\tag{2}
\]

WorldArena 以 \(\mu = -0.1965\)、\(\sigma = 2.3687\) 做 z-score 校准，裁剪至 \([-1.6147, 1.5180]\)，并映射至 \([0.25, 0.75]\)。归一化得分越高，表示相对背景的前景 mask 运动越显著。当 `sample.mask_path` 下存在逐帧分割 mask 时，该指标为 **official**。

<!-- metric-examples:start -->
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和完全一致的展示 prompt，然后比较模型生成结果。paired 展示必须满足 input image 相同、prompt 文本相同，并且一个模型高分、一个模型低分。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `motion_accuracy`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/22ce019a26a7__image_dynamic_photorealistic_articulated_0031.jpg" alt="输入图像: image_dynamic_photorealistic_articulated_0031" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_dynamic_photorealistic_articulated_0031</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. The ibex steps over the sunlit rocky terrain, bending its knees and lowering its curved horns as it navigates patches of grass. The earthy background remains stable.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: yume-sample5b-image-dynamic-5s image_dynamic_photorealistic_articulated_0031">
      <source src="../../../assets/metric_examples/93d9df1827ff__yume-sample5b-image-dynamic-5s__image_dynamic_photorealistic_articulated_0031.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: ltx23-image-dynamic-8gpu-full_20260702_233041 image_dynamic_photorealistic_articulated_0031">
      <source src="../../../assets/metric_examples/8731ae302368__ltx23-image-dynamic-8gpu-full_20260702_233041__image_dynamic_photorealistic_articulated_0031.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/a21cd517f8bc__image_dynamic_photorealistic_articulated_0065.jpg" alt="输入图像: image_dynamic_photorealistic_articulated_0065" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 2 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_dynamic_photorealistic_articulated_0065</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. The elephant steps through savanna grass, swinging its trunk and flapping its ears while sunlight highlights its tusks. Acacia trees and the clear sky remain stable.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: longcat-video-image-dynamic image_dynamic_photorealistic_articulated_0065">
      <source src="../../../assets/metric_examples/a793606886de__longcat-video-image-dynamic__image_dynamic_photorealistic_articulated_0065.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: echo-infinity-image-dynamic image_dynamic_photorealistic_articulated_0065">
      <source src="../../../assets/metric_examples/b55b5fe13514__echo-infinity-image-dynamic__image_dynamic_photorealistic_articulated_0065.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/59c71978624c__image_dynamic_photorealistic_deformable_0152.jpg" alt="输入图像: image_dynamic_photorealistic_deformable_0152" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 3 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_dynamic_photorealistic_deformable_0152</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. The suspended white fabric undulates in slow waves, folding and unfurling along its silky edges. Its overall position stays centered against the blue-gray gradient background.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: echo-infinity-image-dynamic image_dynamic_photorealistic_deformable_0152">
      <source src="../../../assets/metric_examples/13447b8c023e__echo-infinity-image-dynamic__image_dynamic_photorealistic_deformable_0152.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: sana-wm-image-dynamic image_dynamic_photorealistic_deformable_0152">
      <source src="../../../assets/metric_examples/a4db33c6a26d__sana-wm-image-dynamic__image_dynamic_photorealistic_deformable_0152.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/19ee2a11ee26__image_dynamic_photorealistic_rigid_0104.jpg" alt="输入图像: image_dynamic_photorealistic_rigid_0104" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 4 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_dynamic_photorealistic_rigid_0104</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. The helicopter flies low over the calm sea, moving across the open water with its body level. Cargo ships, distant hills, and the clear sky remain steady in the background.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: echo-infinity-image-dynamic image_dynamic_photorealistic_rigid_0104">
      <source src="../../../assets/metric_examples/2030d2058a70__echo-infinity-image-dynamic__image_dynamic_photorealistic_rigid_0104.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: yume-sample5b-image-dynamic-5s image_dynamic_photorealistic_rigid_0104">
      <source src="../../../assets/metric_examples/ba4163c69f03__yume-sample5b-image-dynamic-5s__image_dynamic_photorealistic_rigid_0104.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/66442cb7a3b1__image_dynamic_stylized_articulated_0114.png" alt="输入图像: image_dynamic_stylized_articulated_0114" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 5 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_dynamic_stylized_articulated_0114</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. The adult horse and foal run side by side along the dirt path, legs alternating, manes and tails streaming while fences, farm buildings, hills, and trees remain still.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: yume-sample5b-image-dynamic-5s image_dynamic_stylized_articulated_0114">
      <source src="../../../assets/metric_examples/74b0886de041__yume-sample5b-image-dynamic-5s__image_dynamic_stylized_articulated_0114.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: lingbot_world image_dynamic_stylized_articulated_0114">
      <source src="../../../assets/metric_examples/3d3f5820d0fa__lingbot_world__image_dynamic_stylized_articulated_0114.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/2e07620250c1__image_dynamic_stylized_deformable_0105.png" alt="输入图像: image_dynamic_stylized_deformable_0105" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 6 · 同一输入图像 + 同一 prompt · 分差 1.0000</div>
      <h4><code>image_dynamic_stylized_deformable_0105</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: stylized</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. Wide sheets of fabric spiral around the barren tree, folding, twisting, and tightening into a vortex above the pond. The tree trunk, grass, scattered leaves, and pond edge stay stable.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: echo-infinity-image-dynamic image_dynamic_stylized_deformable_0105">
      <source src="../../../assets/metric_examples/67477bd7cc6c__echo-infinity-image-dynamic__image_dynamic_stylized_deformable_0105.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 1.0000</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: yume-sample5b-image-dynamic-5s image_dynamic_stylized_deformable_0105">
      <source src="../../../assets/metric_examples/00244e911711__yume-sample5b-image-dynamic-5s__image_dynamic_stylized_deformable_0105.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.0000</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "SAM 2 — Segment Anything in Images and Videos"

    Ravi et al., 2024.
    [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

!!! quote "SEA-RAFT — Simple, Efficient, Accurate RAFT for Optical Flow"

    Wang et al., 2024.
    [https://arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
