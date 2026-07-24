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
## 代表性同 Case 对比
每个展示块先给出该 case 的 input image 和完全一致的展示 prompt，然后比较模型生成结果。paired 展示必须满足 input image 相同、prompt 文本相同，并且一个模型高分、一个模型低分。若当前正式快照中该 metric 没有这种跨模型同 case pair，则单独展示高分/低分极值样例，不把它们伪装成同 case 对比。
机器可读 paired 导出： `docs/assets/metric_examples/paired_manifest.csv`.
### `semantic_alignment`
<div class="metric-case-stack">
<article class="metric-case-exhibit">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="../../../assets/metric_examples/inputs/fe16fdec2c52__image_dynamic_photorealistic_articulated_0001.jpg" alt="输入图像: image_dynamic_photorealistic_articulated_0001" loading="lazy" decoding="async">
      <figcaption>输入图像</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">Case 1 · 同一输入图像 + 同一 prompt · 分差 0.0678</div>
      <h4><code>image_dynamic_photorealistic_articulated_0001</code></h4>
      <div class="metric-case-meta"><span>Suite: image_dynamic</span><span>Scene: photorealistic</span></div>
      <div class="metric-case-prompt">
        <span>目标 prompt</span>
        <p>Camera fixed. A bird of prey glides forward low over a grassy field, wings spread wide with subtle body and wing adjustments. The wooden fence post, wild grasses, and vines remain still.</p>
        <small>prompt 来源: prompt_target</small>
      </div>
    </div>
  </div>
  <div class="metric-output-grid">
<figure class="metric-output-card metric-output-card--good">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="好结果: _readiness_echo_dynamic_smoke_after image_dynamic_photorealistic_articulated_0001">
      <source src="../../../assets/metric_examples/8f778faac6c6___readiness_echo_dynamic_smoke_after__image_dynamic_photorealistic_articulated_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--good">好结果 · 分数 0.6668</div>
  </div>
</figure>
<figure class="metric-output-card metric-output-card--bad">
  <div class="metric-output-media-wrap">
    <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="差结果: stable-virtual-camera-image-dynamic-5s image_dynamic_photorealistic_articulated_0001">
      <source src="../../../assets/metric_examples/73ddbcad1071__stable-virtual-camera-image-dynamic-5s__image_dynamic_photorealistic_articulated_0001.mp4" type="video/mp4">
    </video>
    <div class="metric-score-pill metric-score-pill--bad">差结果 · 分数 0.5990</div>
  </div>
</figure>
  </div>
</article>
</div>
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "CLIP — Learning Transferable Visual Models From Natural Language Supervision"

    Radford et al., ICML 2021.
    [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
