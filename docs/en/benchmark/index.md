# Benchmark Overview

## Task formulation

Grounded in our operational definition of world models and the formal criterion introduced in Sec. 2, WorldAtlas Arena formulates world generation as a set of conditional next-world generation tasks across video, 3D, and 4D modalities.

### Task specification

Following a next-scene evaluation perspective, each task instance is specified by a tuple \((C, N, L)\):

- **\(C = \{X, P\}\)** — current evidence: an image or video observation \(X\) and an optional textual description \(P\)
- **\(N\)** — target world specification: the next scene, dynamic change, object transformation, or interaction outcome to be generated
- **\(L\)** — layout and control specification: camera trajectory, viewpoint change, navigation path, object-level control, action instruction, or fixed-camera constraint

### Generative interface

Given this specification, a model is required to generate an output artifact

\[
O = g_{\mathrm{world}}(w_{\mathrm{proc}}(C, N, L)),
\]

where:

- \(g_{\mathrm{world}}\) denotes the evaluated generative system
- \(w_{\mathrm{proc}}\) denotes model-specific preprocessing
- \(O\) may be a video sequence, a 3D scene, a 4D dynamic asset, or an interaction rollout depending on the task setting

This formulation provides a common interface for evaluating heterogeneous systems while allowing each modality to retain its own output structure and control semantics.

### Task regimes

The task suite separates world generation into complementary regimes that probe different capabilities:

- **Static world generation** — extends or transforms a scene under camera movement, viewpoint change, or scene-level layout specification while preserving geometry, appearance, and semantic consistency
- **Dynamic world generation** — in-scene temporal evolution (object motion, human activity, environmental dynamics, state transitions), often under fixed or controlled camera settings to isolate dynamics from viewpoint changes
- **Interaction-conditioned generation** — generated futures remain consistent with explicit actions, object controls, or intervention instructions

By decomposing tasks in this way, WorldAtlas Arena distinguishes perceptual quality from spatial consistency, temporal dynamics, and controllability, enabling systematic comparison across video generators, 3D world generators, 4D dynamic scene models, and action-conditioned world models.

## Detailed Algorithm {#detailed-algorithm}

### Problem formulation

WorldAtlas Arena treats each benchmark instance as a conditional next-world generation problem. A task is specified by the tuple \((C, N, L)\), and an evaluated system produces a generated world observation \(O\). The evaluation objective is to quantify how well \(O\) satisfies the contract implied by \((C, N, L)\) across complementary capability dimensions—perceptual quality, spatial and temporal consistency, controllability, physical plausibility, and real-time behavior.

Formally, for sample \(i\) and metric \(j\), the benchmark seeks normalized scores

\[
\tilde{s}_{i,j} \in [0, 1],
\]

together with suite-level and dimension-level aggregates over eligible outputs. No single benchmark-wide scalar is defined; comparability requires matched task regimes, sampling protocol, and prediction completeness.

### Inputs and outputs

**Inputs.**

| Symbol | Role |
| --- | --- |
| \(C = \{X, P\}\) | Current evidence: visual observation \(X\) (image or video) and optional textual description \(P\) |
| \(N\) | Navigation or target-world specification: the intended future scene, dynamic change, or interaction outcome |
| \(L\) | Layout and control specification: camera trajectory, viewpoint change, navigation path, object-level control, action instruction, or fixed-camera constraint |
| \(O\) | Generated world observation produced by the evaluated generative system |

Auxiliary supervision—reference rollouts, pose trajectories, segmentation masks, or instruction labels—may accompany a sample when available. These sidecars gate which metrics are applicable but are not part of the core generative contract.

**Outputs.**

| Output | Description |
| --- | --- |
| Per-sample metric scores | Raw measurements and their normalized counterparts for each applicable metric |
| Eligibility status | Whether a metric enters headline aggregation, is recorded as non-official, or is not applicable for that sample |
| Aggregated benchmark scores | Suite-level per-metric averages and dimension-level roll-ups computed only over eligible official scores |
| Diagnostic summaries | Separate aggregates for analysis-only metrics that do not enter headline comparison |

### Evaluation pipeline

The benchmark applies a fixed five-stage procedure to every evaluated sample.

**Stage 1 — Task instantiation.** Each sample materializes \((C, N, L)\) under a suite-specific conditioning regime (static world generation, dynamic world generation, or interaction-conditioned generation). The conditioning contract determines which portion of the evidence serves as input and which temporal region of the rollout is evaluated.

**Stage 2 — Protocol alignment.** A deterministic temporal sampling protocol selects matched evaluation support on the prediction and, when required, on the reference stream. Anchor positions and motion windows ensure that prediction–reference comparisons occur on aligned time indices rather than arbitrary frame offsets.

**Stage 3 — Metric computation.** For each configured metric \(m_j\), an evaluator computes a raw measurement

\[
r_{i,j} = f_j\!\left(O_i,\, C_i,\, \mathcal{A}_i\right),
\]

where \(\mathcal{A}_i\) denotes the auxiliary supervision available for sample \(i\). Metrics span quality, spatial consistency, temporal dynamics, controllability, physics realism, and real-time calibration. Some metrics operate on \(O\) alone; others require paired reference content or annotated sidecars.

**Stage 4 — Normalization and eligibility.** Raw measurements are transformed into normalized scores \(\tilde{s}_{i,j} \in [0, 1]\) via bounded scaling, empirical calibration, or multi-component reduction. Each output is tagged according to sample-level supervision gates: official scores enter headline aggregation; non-official scores are reported separately; ineligible metrics are excluded rather than imputed.

**Stage 5 — Aggregation.** Official normalized scores are averaged within suites to form per-metric summaries, then regrouped into the six evaluation dimensions. Diagnostic metrics are aggregated on a parallel track. Missing predictions and ineligible metrics are excluded from averages—they are never treated as zero.

### Design principles

The pipeline exposes a common \((C, N, L) \rightarrow O\) interface for heterogeneous generative systems while allowing each modality to retain its own output structure and control semantics. Decomposing tasks into static, dynamic, and interaction-conditioned regimes isolates distinct failure modes. Separating official, non-official, and not-applicable channels preserves interpretability: a model is compared only on metrics for which the sample provides sufficient supervision and the system produced a valid observation.

## Simulator-Backed Physics

`config/benchmark_physics.yaml` defines a simulator-backed physics configuration under track `physics_i2v`. Its results stay outside the four-suite public vectors and should not be mixed with them. See [Metrics / Physics](metrics/dimensions/physics.md) and the individual simulator-backed metric group pages linked from that dimension overview.
