# Camera Error

Camera Error 量化从生成视频恢复的相机路径与标注 ground-truth 外参之间的**轨迹一致程度**，评估视点控制合规性，而非物体运动或指令跟随。

在 direct pose-GT 策略下，记 \(\{\mathbf{T}^{\mathrm{gt}}_t\}_{t=1}^{N}\) 与 \(\{\mathbf{T}^{\mathrm{pred}}_t\}_{t=1}^{N}\) 分别为标注与视觉里程计（MegaSAM 或 DROID-SLAM）得到的 $4 \times 4$ camera-to-world 矩阵。若两者帧数不同，先通过平移线性插值和四元数 SLERP 将 GT 均匀重采样到恢复出的 pose 数量，再求完整的正尺度相似变换：

\[
s^\star,\mathbf{Q}^\star,\mathbf{u}^\star
= \arg\min_{s>0,\,\mathbf{Q}\in SO(3),\,\mathbf{u}}
\sum_{t=1}^{N}
\left\|\mathbf{t}^{\mathrm{gt}}_t-
\left(s\mathbf{Q}\mathbf{t}^{\mathrm{pred}}_t+\mathbf{u}\right)\right\|_2^2
\tag{1}
\]

该变换使用 Umeyama 对齐；对于静止或共线轨迹，使用对应相机朝向消除旋转自由度，再求尺度和平移。对齐后 \(\tilde{\mathbf{R}}_t=\mathbf{Q}^\star\mathbf{R}^{\mathrm{pred}}_t\)，\(\tilde{\mathbf{t}}_t=s^\star\mathbf{Q}^\star\mathbf{t}^{\mathrm{pred}}_t+\mathbf{u}^\star\)。平均平移误差为

\[
E_t = \frac{1}{N}\sum_{t=1}^{N} \left\|\tilde{\mathbf{t}}_t - \mathbf{t}^{\mathrm{gt}}_t\right\|_2
\tag{2}
\]

平均旋转误差（度）由相对旋转 \(\mathbf{R}^{\mathrm{rel}}_t = \tilde{\mathbf{R}}_t (\mathbf{R}^{\mathrm{gt}}_t)^\top\) 给出：

\[
E_r = \frac{1}{N}\sum_{t=1}^{N} \arccos\!\left(\mathrm{clip}\!\left(\frac{\mathrm{tr}(\mathbf{R}^{\mathrm{rel}}_t) - 1}{2}, -1, 1\right)\right) \cdot \frac{180}{\pi}
\tag{3}
\]

WorldArena 将 direct-GT 的两个分量按经验范围 \([0, 15°]\)（旋转）与 \([0, 0.5]\)（平移）归一化至 \([0,1]\)（越低越好），再取**几何均值**。默认的 `native_intended` MegaSAM 策略则把恢复出的运动与请求的相机/动作方向比较，直接返回其 direction-confidence 分数，不使用上述 direct-GT reducer。所选策略具备足够相机监督时，该指标为 **official**。

上述 direct-GT 经验区间早于 `sim3_v1`，发布修正后的 direct-GT 归一化分数前必须重新验证。当前默认的 `native_intended` 分数面不受此影响。

<!-- metric-examples:start -->
## 暂无可核验的同 Case 对比

> 当前保留的 `artifacts/reports` 快照中，没有同一 case、同一 Camera Error 协议下的两条不同预测。旧卡片实际是同一视频经过不同 backend 的结果，因此已移除，不再伪装成好/差输出对照。
<!-- metric-examples:end -->

## 参考文献 {#reference}

!!! quote "DROID-SLAM — Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras"

    Teed & Deng, NeurIPS 2021.
    [https://arxiv.org/abs/2108.10869](https://arxiv.org/abs/2108.10869)
