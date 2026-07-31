# Depth Collision

Depth Collision 是二值标记，检测相机正前方是否出现不合理近距几何。Video Depth Anything 提供逐帧 metric 深度；中心裁剪检验 rollout 过程中前向视廊是否异常逼近。

对帧 \(t\) 的深度图 \(D_t\) 与中心裁剪 \(\mathcal{C}\)：

\[
\bar{d}_t = \frac{1}{|\mathcal{C}|}\sum_{(x,y)\in\mathcal{C}} D_t(x,y)
\tag{1}
\]

给定阈值 \(\tau\)（默认 \(1.0\)），碰撞指示量为

\[
e_{\mathrm{collision}} = \mathbf{1}\!\left[\min_t \bar{d}_t < \tau\right]
\tag{2}
\]

原始得分 \(e_{\mathrm{collision}} \in \{0,1\}\) 经反向归一化，无碰撞对应更高得分。该标记在估计深度下代理近距失败（穿墙、贴脸运动），并非物理引擎接触判定。

<!-- metric-examples:start -->
## 暂无可核验的同 Case 对比

> 当前保留的正式报告只覆盖该指标的一条 prediction run，无法组成经过核验的同-case高/低分输出对。这里刻意不再展示不同 case 的极值。
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"

    Chen et al., arXiv:2501.12375, 2025.
