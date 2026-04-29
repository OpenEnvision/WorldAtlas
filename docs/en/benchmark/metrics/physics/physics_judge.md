# Physics Judge Metrics

The `physics_judge` metric family comes from `config/benchmark_physics_judge.yaml`. It does not depend on simulator world artifacts. Instead, it samples prediction frames, optionally samples reference frames, and asks a VLM judge to produce `0..1` scores against the physical rubric stored in `physics_spec.judge_card`.

The default backend is `openai_judge`, and `gemini_judge` is also supported. Higher is better for every judge metric.

## Inputs

| Input | Use |
| --- | --- |
| prediction video frames | primary object being scored |
| reference frames | optional context for the target phenomenon and salient motion pattern |
| `physics_spec.judge_card` | rubric, physics family, expected pattern, forbidden failures, and weights |
| prompt fields | context such as `prompt_current`, `prompt_target`, and motion category |

## Formulas

The judge backend expects the provider JSON to contain five raw sub-scores:

\[
\hat{s}_{phenomenon},\quad
\hat{s}_{causality},\quad
\hat{s}_{material},\quad
\hat{s}_{temporal},\quad
\hat{s}_{prompt}
\]

The implementation clamps each score to \([0,1]\):

\[
s_d=\min(1,\max(0,\hat{s}_d))
\]

Therefore:

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

Overall uses the judge-card weights. Without per-sample overrides, defaults are:

\[
w_{phenomenon}=0.25,\quad
w_{causality}=0.25,\quad
w_{material}=0.20,\quad
w_{temporal}=0.20,\quad
w_{prompt}=0.10
\]

Only nonnegative weights are used:

\[
w^+_d=\max(w_d,0)
\]

\[
\mathrm{physics\_judge\_overall}
=
\frac{\sum_d w^+_d s_d}
{\sum_d w^+_d}
\]

If \(\sum_d w^+_d=0\), the implementation uses denominator \(1\); the numerator is also \(0\), so overall is \(0\).

## Metrics

| Metric | Judged dimension | Raw score meaning | Current normalization |
| --- | --- | --- | --- |
| `physics_judge_phenomenon` | phenomenon presence | Whether the requested physical phenomenon is clearly visible, such as acceleration, collision, fluid flow, reflection, or phase change. | `[0.0, 1.0]` |
| `physics_judge_causality` | causal and dynamic direction | Whether motion, force, contact, rebound, flow direction, or lighting changes follow plausible causal structure instead of unexplained jumps or reversals. | `[0.0, 1.0]` |
| `physics_judge_material` | material and shape behavior | Whether rigidity, deformation, fluid topology, phase state, reflection/refraction medium behavior, and related material cues match the physical category. | `[0.0, 1.0]` |
| `physics_judge_temporal` | temporal continuity | Whether the phenomenon evolves continuously over time and avoids teleportation, flicker, sudden disappearance, geometry popping, or other impossible discontinuities. | `[0.0, 1.0]` |
| `physics_judge_prompt` | prompt adherence | Whether the generated video still follows the sample prompt, target subject, scene, and requested physical behavior. | `[0.0, 1.0]` |
| `physics_judge_overall` | weighted total | Weighted average of the five sub-scores above. Default weights are phenomenon `0.25`, causality `0.25`, material `0.20`, temporal `0.20`, and prompt `0.10`. | `[0.0, 1.0]` |

Per-sample `judge_card.weights` may override the default weights. `physics_judge_overall` is computed after re-normalizing nonnegative weights.

## Judge Card

If a sample already carries `physics_spec.judge_card`, the backend uses it directly. Otherwise it builds a rubric card from group, dimension, caption, and reference label. A card usually contains:

- `family`: for example `kinematics`, `rigid_dynamics`, `deformation`, `fluid`, `phase_change`, `reflection`, `refraction`, `scattering`, or `lighting`
- `expected_pattern`: the motion or physical phenomenon expected for this physics dimension
- dimension-specific questions: phenomenon presence, causal direction, material behavior, temporal continuity
- `forbidden_failures`: examples include teleportation, phase-through, surface flicker, and mass disappearance
- `weights`: overall-score weights for the five sub-scores

## Details Payload

Each judge metric records:

- `provider` and `model`
- `judge_card`
- `confidence`
- `failures`
- `reasoning`
- `cache_hit`
- `success`
- possibly `provider_response_text` in debug-style configurations

## Good For

- Visually scoring physics plausibility when simulator artifacts are incomplete.
- Covering phenomena that low-level geometry metrics do not fully capture, such as non-rigid motion, fluids, phase changes, reflection, and refraction.
- Checking prompt adherence and physical behavior together.

## Not Enough For

- Judge scores are model judgments, not deterministic geometric measurements.
- Scores depend on frame sampling, reference frames, VLM capability, and rubric wording.
- They should not replace simulator-backed core metrics; report them separately or as supporting evidence.
