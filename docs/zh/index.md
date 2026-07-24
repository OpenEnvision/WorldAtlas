# WorldAtlas Arena

WorldAtlas Arena 由 [OpenEnvision](https://openenvision.github.io/index.html) 开发维护——一个面向开放视觉智能与世界建模的产学研联合研究平台。

## Summary

WorldAtlas Arena 围绕分级能力体系——从感知与持久场景建模，到操控、动力学与闭环交互——而非单纯的表面保真度，来评测生成式世界模型。

## Capability Gradient

对世界模型的评估常常依赖输出的视觉可信度，但这一标准可能掩盖了系统在底层能力上的本质差异。更具信息量的视角是将这些能力组织为一个分级体系：

- **感知（perception）** — 从观测中还原已经存在的信息
- **建模（modeling）** — 维持超越单帧观测的持久内部状态
- **操控（manipulation）** — 在显式约束下对该状态进行受控变换
- **动力学（dynamics）** — 使状态依照物理与因果规律随时间演化
- **交互（interaction）** — 生成兼具状态持久性与反事实有效性的、以动作为条件的推演

较低层级的能力并不蕴含更高层级的能力。

WorldAtlas Arena 正是围绕这一能力梯度而非单纯的表面保真度来构建评测：其评测套件与指标旨在判定被测系统实际达到的能力层级，覆盖从感知与场景建模，到物理一致的动力学与闭环交互的全谱任务。仓库的组织方式也体现了这种关注点分离——评测逻辑、适配器契约、第三方仓库与缓存 checkpoint 被设计为可以彼此独立演进。

## 文档导航

如果需要了解正式的基准评测框架与任务分解，请从[基准评测总览](benchmark/index.md)开始。其余文档的组织方式如下：

<div class="doc-map">
<div class="doc-map-grid">
<a class="doc-map-card doc-map-card--benchmark" href="benchmark/">
  <span class="doc-map-card-kicker">01 · 评测</span>
  <span class="doc-map-card-title">基准评测</span>
  <span class="doc-map-card-stat">4 个 suite</span>
  <span class="doc-map-card-desc">评测范围、套件、维度和报告契约。</span>
</a>
<a class="doc-map-card doc-map-card--metrics" href="benchmark/metrics/">
  <span class="doc-map-card-kicker">02 · 指标</span>
  <span class="doc-map-card-title">指标</span>
  <span class="doc-map-card-stat">39 篇文档</span>
  <span class="doc-map-card-desc">按评测维度分组的完整指标体系。</span>
</a>
<a class="doc-map-card doc-map-card--runtime" href="thirdparty/">
  <span class="doc-map-card-kicker">03 · 运行时</span>
  <span class="doc-map-card-title">运行时布局</span>
  <span class="doc-map-card-stat">适配器</span>
  <span class="doc-map-card-desc">第三方仓库、checkpoint 和适配器接线方式。</span>
</a>
<a class="doc-map-card doc-map-card--dataset" href="overview/">
  <span class="doc-map-card-kicker">04 · 语料</span>
  <span class="doc-map-card-title">数据分析</span>
  <span class="doc-map-card-stat">11,653 条</span>
  <span class="doc-map-card-desc">集合规模、标注覆盖率和媒体统计信息。</span>
</a>
</div>
</div>

!!! tip "推荐第一步"
    建议先阅读 [基准评测总览](benchmark/index.md) 和 `config/models/` 下的模型配置，它们定义了本仓库公开的核心契约。
