# 深度碰撞

`depth_collision` 用 Video Depth Anything [[Chen et al., 2025](#reference)] 的 metric depth 检测生成 rollout 是否出现“相机撞到近处物体”的迹象。
它关注的是相机前方中心区域是否在生成过程中突然变得过近，从而暴露出穿墙、撞物、贴脸推进等 camera-path failure。

## 它测什么

- 相机路径中心区域是否突然贴近前景或障碍物。
- 生成视频是否出现穿墙、撞物、贴脸推进这类 camera-path failure。
- rollout 过程中视野前方是否存在异常近距离深度。

## 原理说明

生成视频中的相机运动如果不稳定，常见现象是相机轨迹穿入物体、贴近墙面、钻入前景，或者在 rollout 中突然推进到不可见的近距离区域。
这些错误未必能通过单帧图像质量直接发现，但会在深度图上表现为视野中心区域的估计深度异常变小。

`depth_collision` 因此先使用 Video Depth Anything 的 metric depth 模式为每一帧预测深度。
Video Depth Anything 面向长视频一致深度估计，适合在连续帧上提供相对稳定的逐帧深度信号。
随后，指标只检查深度图中心的固定比例窗口，因为相机运动方向通常接近画面中心，中心区域更能反映“相机前方是否出现近距离障碍”。

如果任意一帧的中心窗口平均深度低于阈值，指标就认为该 rollout 触发了 depth collision。
这里的 collision 表示“估计深度意义上的近距告警”，不是通过三维碰撞体、物理引擎或真实接触标注得到的碰撞事件。

## 当前算法

1. 使用 `Video-Depth-Anything` [[Chen et al., 2025](#reference)] metric 模式预测每帧深度。
2. 默认读取 benchmark 已采样帧；`frame_source = full_video` 时读取完整预测视频。
3. 在每帧深度图中心裁一个固定比例窗口，默认 `center_crop_ratio = 0.2`。
4. 计算中心窗口平均 metric depth：`center_depth_t = mean(depth_t[center_crop])`。
5. 若任一帧 `center_depth_t < depth_threshold`，默认阈值 `1.0`，判定 collision。
6. 原始分数是二值 flag：`raw = 1.0` 表示发生 collision，`raw = 0.0` 表示未检测到 collision。
7. 归一化使用反向缩放：`raw = 0` 映射为 `normalized = 1`，`raw = 1` 映射为 `normalized = 0`。

## 误差公式

对第 \(t\) 帧，Video Depth Anything 预测出的深度图记为 \(D_t\)，中心裁剪区域记为 \(\mathcal{C}\)。
该帧的中心区域平均深度为：

!!! info "中心深度"

    \[
    \bar{d}_t
    =
    \frac{1}{|\mathcal{C}|}
    \sum_{(x,y)\in\mathcal{C}}
    D_t(x,y)
    \]

给定深度阈值 \(\tau\)，单个视频的 depth collision flag 定义为：

!!! info "碰撞判定"

    \[
    e_{\mathrm{collision}}
    =
    \mathbb{1}
    \left[
        \min_t \bar{d}_t < \tau
    \right]
    \]

其中，\(\mathbb{1}[\cdot]\) 是指示函数；当任意一帧的中心平均深度低于阈值时，\(e_{\mathrm{collision}}=1\)，表示检测到近距碰撞风险；否则 \(e_{\mathrm{collision}}=0\)。
当前默认阈值为 \(\tau=1.0\)，默认中心裁剪比例为 `0.2`。

## 输入与输出

| 项目 | 当前实现 |
| --- | --- |
| 预测输入 | prediction video 的采样帧或完整帧 |
| 参考输入 | 不使用 |
| 标注输入 | 不使用 |
| 后端 | `video_depth_anything` |
| 运行时默认 | `repo_root = thirdparty/Video-Depth-Anything`、`checkpoint_path = ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth`、`encoder = vits`、`input_size = 256` |
| 原始输出 | `0.0` / `1.0` collision flag，越低越好 |
| 归一化输出 | `[0,1]`，越高越好 |
| 细节字段 | `center_crop_mean_depths`、`collision`、`first_collision_frame`、`threshold_results`、`depth_shape`、`frame_source`、`frame_indices` |


## 参考文献 {#reference}

!!! quote "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"

    Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang.  
    arXiv:2501.12375, 2025.

??? note "BibTeX"

    ```bibtex
    @article{video_depth_anything,
        title = {Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
        author = {Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi},
        journal = {arXiv:2501.12375},
        year = {2025},
    }
    ```
