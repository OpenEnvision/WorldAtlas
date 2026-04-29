# Physics Judge Metrics

`physics_judge` 指标族来自 `config/benchmark_physics_judge.yaml`。它不依赖 simulator world artifact，而是抽取 prediction frames，并可选抽取 reference frames，让 VLM judge 按 `physics_spec.judge_card` 中的物理 rubric 输出 `0..1` 分数。

默认 backend 是 `openai_judge`，也支持 `gemini_judge`。所有 judge 指标都是越高越好。

## 输入

| 输入 | 用途 |
| --- | --- |
| prediction video frames | judge 的主要评分对象 |
| reference frames | 可选，用来理解目标物理现象和关键运动模式 |
| `physics_spec.judge_card` | rubric、物理类别、expected pattern、禁止失败模式和权重 |
| prompt fields | `prompt_current`、`prompt_target`、motion category 等上下文 |

## 公式

judge backend 要求 provider 返回 JSON 中的五个 raw sub-score：

\[
\hat{s}_{phenomenon},\quad
\hat{s}_{causality},\quad
\hat{s}_{material},\quad
\hat{s}_{temporal},\quad
\hat{s}_{prompt}
\]

实现会把每个分数 clamp 到 \([0,1]\)：

\[
s_d=\min(1,\max(0,\hat{s}_d))
\]

于是：

\[
\mathrm{physics\_judge\_phenomenon}=s_{phenomenon}
\]

\[
\mathrm{physics\_judge\_causality}=s_{causality}
\]

\[
\mathrm{physics\_judge\_material}=s_{material}
\]

\[
\mathrm{physics\_judge\_temporal}=s_{temporal}
\]

\[
\mathrm{physics\_judge\_prompt}=s_{prompt}
\]

overall 使用 judge card 中的权重；若样本没有覆盖，默认：

\[
w_{phenomenon}=0.25,\quad
w_{causality}=0.25,\quad
w_{material}=0.20,\quad
w_{temporal}=0.20,\quad
w_{prompt}=0.10
\]

实现只使用非负权重：

\[
w^+_d=\max(w_d,0)
\]

\[
\mathrm{physics\_judge\_overall}
=
\frac{\sum_d w^+_d s_d}
{\sum_d w^+_d}
\]

如果 \(\sum_d w^+_d=0\)，实现用 denominator \(1\)，此时 numerator 也是 \(0\)，overall 为 \(0\)。

## 指标

| 指标 | 评分维度 | 原始分数含义 | 当前归一化 |
| --- | --- | --- | --- |
| `physics_judge_phenomenon` | 现象是否出现 | 目标物理现象是否清楚可见，例如加速、碰撞、液体流动、反射或相变是否真的发生。 | `[0.0, 1.0]` |
| `physics_judge_causality` | 因果与动力学方向 | 运动、力、接触、反弹、流向、光照变化等是否遵循合理因果关系，而不是无原因跳变或反向演化。 | `[0.0, 1.0]` |
| `physics_judge_material` | 材料与形态行为 | 物体刚性、形变、流体拓扑、相态、反射/折射介质等是否与对应物理类别一致。 | `[0.0, 1.0]` |
| `physics_judge_temporal` | 时间连续性 | 现象是否随时间连续演化，是否避免 teleportation、闪烁、突然消失、几何弹跳等不可能的时序断裂。 | `[0.0, 1.0]` |
| `physics_judge_prompt` | prompt 一致性 | 生成视频是否仍然符合样本 prompt、目标主体、场景和 requested physical behavior。 | `[0.0, 1.0]` |
| `physics_judge_overall` | 加权总分 | 上面五个子分数的加权平均。默认权重是 phenomenon `0.25`、causality `0.25`、material `0.20`、temporal `0.20`、prompt `0.10`。 | `[0.0, 1.0]` |

具体样本的 `judge_card.weights` 可以覆盖默认权重。`physics_judge_overall` 会按非负权重重新归一化后计算。

## Judge Card

如果样本已经带有 `physics_spec.judge_card`，backend 会直接使用。否则会根据 group、dimension、caption 和 reference label 生成 rubric card。card 中通常包含：

- `family`：例如 `kinematics`、`rigid_dynamics`、`deformation`、`fluid`、`phase_change`、`reflection`、`refraction`、`scattering`、`lighting`
- `expected_pattern`：该 physics dimension 期待出现的运动或物理现象
- dimension-specific questions：现象出现、因果方向、材料行为、时间连续性
- `forbidden_failures`：例如 teleportation、phase-through、surface flicker、mass disappearance
- `weights`：五个子分数的 overall 权重

## 输出细节

每个 judge metric 的 `details` 会记录：

- `provider` 与 `model`
- `judge_card`
- `confidence`
- `failures`
- `reasoning`
- `cache_hit`
- `success`
- 调试配置下可能包含 `provider_response_text`

## 适合解读的问题

- simulator artifact 不完整时，是否仍能用视觉 rubric 粗评物理合理性。
- 非刚体、流体、相变、反射/折射等难以完全用低层几何指标覆盖的现象。
- prompt adherence 与物理现象是否同时成立。

## 不适合单独解读的问题

- judge 分数是模型判断，不是确定性几何测量。
- 评分会受抽帧数量、reference frame、VLM 能力和 rubric 表述影响。
- 它不能替代 simulator-backed core metric，只适合与低层指标分开报告或作为补充。
