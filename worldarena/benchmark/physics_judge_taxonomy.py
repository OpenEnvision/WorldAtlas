from __future__ import annotations

from copy import deepcopy
from typing import Any


_FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "kinematics": {
        "family": "kinematics",
        "phenomenon_presence_question": "Is the requested motion pattern clearly present?",
        "causal_direction_question": "Does the motion follow a physically plausible direction or trend?",
        "material_behavior_question": "Does the object preserve plausible shape and contact behavior?",
        "temporal_continuity_question": "Is the motion temporally continuous without teleportation or abrupt impossible jumps?",
        "forbidden_failures": [
            "teleportation",
            "direction_reversal_without_cause",
            "shape_instability",
            "contact_violation",
        ],
        "weights": {
            "phenomenon": 0.3,
            "causality": 0.3,
            "material": 0.15,
            "temporal": 0.15,
            "prompt": 0.1,
        },
    },
    "rigid_dynamics": {
        "family": "rigid_dynamics",
        "phenomenon_presence_question": "Is the rigid-body interaction or motion clearly visible?",
        "causal_direction_question": "Do forces, impacts, rebounds, or rotations evolve in a plausible way?",
        "material_behavior_question": "Does the object remain rigid except where the category allows deformation?",
        "temporal_continuity_question": "Are trajectories and post-contact outcomes continuous over time?",
        "forbidden_failures": [
            "teleportation",
            "phase_through_objects",
            "energy_gain_without_cause",
            "geometry_flicker",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.3,
            "material": 0.2,
            "temporal": 0.15,
            "prompt": 0.1,
        },
    },
    "deformation": {
        "family": "deformation",
        "phenomenon_presence_question": "Is non-rigid deformation or size change clearly visible?",
        "causal_direction_question": "Does the change happen for a plausible reason such as force, inflation, or heating?",
        "material_behavior_question": "Does the material change shape in a coherent way instead of fragmenting randomly?",
        "temporal_continuity_question": "Does the non-rigid change evolve smoothly over time?",
        "forbidden_failures": [
            "shape_pop",
            "mass_disappearance",
            "surface_flicker",
            "causeless_change",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.25,
            "material": 0.25,
            "temporal": 0.15,
            "prompt": 0.1,
        },
    },
    "fluid": {
        "family": "fluid",
        "phenomenon_presence_question": "Is fluid or gas-like flow clearly visible?",
        "causal_direction_question": "Does the motion follow plausible gravity, pressure, or flow direction cues?",
        "material_behavior_question": "Does the material keep coherent fluid topology instead of behaving like rigid chunks?",
        "temporal_continuity_question": "Is the flow temporally coherent without sudden discontinuities?",
        "forbidden_failures": [
            "teleportation",
            "rigid_chunk_motion",
            "mass_disappearance",
            "flow_direction_without_cause",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.25,
            "material": 0.25,
            "temporal": 0.15,
            "prompt": 0.1,
        },
    },
    "phase_change": {
        "family": "phase_change",
        "phenomenon_presence_question": "Is the phase transition clearly visible?",
        "causal_direction_question": "Does the transition happen in a plausible thermodynamic direction?",
        "material_behavior_question": "Do texture, volume, and material state evolve coherently?",
        "temporal_continuity_question": "Does the state transition evolve gradually and continuously?",
        "forbidden_failures": [
            "instant_full_transition",
            "state_flicker",
            "mass_disappearance",
            "causeless_reversal",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.25,
            "material": 0.25,
            "temporal": 0.15,
            "prompt": 0.1,
        },
    },
    "reflection": {
        "family": "reflection",
        "phenomenon_presence_question": "Is a reflection clearly visible?",
        "causal_direction_question": "Does the reflected image change consistently with the source scene or object?",
        "material_behavior_question": "Does the reflective surface remain plausible and not behave like a second independent object?",
        "temporal_continuity_question": "Is the reflection temporally stable and coupled to scene motion?",
        "forbidden_failures": [
            "reflection_decoupling",
            "missing_reflected_motion",
            "double_object_independence",
            "surface_flicker",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.3,
            "material": 0.15,
            "temporal": 0.2,
            "prompt": 0.1,
        },
    },
    "refraction": {
        "family": "refraction",
        "phenomenon_presence_question": "Is refractive distortion clearly visible?",
        "causal_direction_question": "Does the distorted background remain plausibly tied to transparent media?",
        "material_behavior_question": "Does the transparent material remain coherent rather than turning into arbitrary texture noise?",
        "temporal_continuity_question": "Is the distortion stable over time as objects or camera move?",
        "forbidden_failures": [
            "background_decoupling",
            "distortion_pop",
            "texture_noise_without_medium",
            "surface_flicker",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.3,
            "material": 0.15,
            "temporal": 0.2,
            "prompt": 0.1,
        },
    },
    "scattering": {
        "family": "scattering",
        "phenomenon_presence_question": "Is volumetric or atmospheric scattering clearly visible?",
        "causal_direction_question": "Do beams, haze, or depth cues react plausibly to light and scene layout?",
        "material_behavior_question": "Does the medium behave coherently as atmosphere, fog, mist, or smoke?",
        "temporal_continuity_question": "Is the scattering effect stable over time instead of flickering arbitrarily?",
        "forbidden_failures": [
            "beam_pop",
            "lighting_decoupling",
            "depth_inconsistency",
            "temporal_flicker",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.3,
            "material": 0.15,
            "temporal": 0.2,
            "prompt": 0.1,
        },
    },
    "lighting": {
        "family": "lighting",
        "phenomenon_presence_question": "Is the lighting effect clearly visible?",
        "causal_direction_question": "Do highlights, shadows, blooms, or beams respond plausibly to light sources?",
        "material_behavior_question": "Do scene surfaces and atmospherics react coherently to the light?",
        "temporal_continuity_question": "Is the lighting effect stable over time without implausible flicker?",
        "forbidden_failures": [
            "shadow_direction_inconsistency",
            "highlight_pop",
            "bloom_flicker",
            "light_source_decoupling",
        ],
        "weights": {
            "phenomenon": 0.25,
            "causality": 0.3,
            "material": 0.15,
            "temporal": 0.2,
            "prompt": 0.1,
        },
    },
}


_DIMENSION_OVERRIDES: dict[str, dict[str, Any]] = {
    "acceleration": {
        "family": "kinematics",
        "expected_pattern": "speed should increase over time or during takeoff / thrust phases",
    },
    "deceleration": {
        "family": "kinematics",
        "expected_pattern": "speed should decrease over time during braking or landing-like motion",
    },
    "uniform_motion": {
        "family": "kinematics",
        "expected_pattern": "velocity should remain approximately steady without large unexplained changes",
    },
    "motion_3d": {
        "family": "kinematics",
        "expected_pattern": "the main subject should move through depth, not only via camera motion",
    },
    "slope_sliding": {
        "family": "rigid_dynamics",
        "expected_pattern": "the object should move down the slope with plausible contact",
    },
    "parabolic_motion": {
        "family": "kinematics",
        "expected_pattern": "trajectory should resemble a projectile arc under gravity",
    },
    "parabolic_motion_with_rotation": {
        "family": "rigid_dynamics",
        "expected_pattern": "the tossed object should follow an arc while rotating continuously",
    },
    "circular_motion": {
        "family": "kinematics",
        "expected_pattern": "the object should move around a center instead of only spinning in place",
    },
    "rotation": {
        "family": "rigid_dynamics",
        "expected_pattern": "the rigid object should rotate with stable geometry and angular continuity",
    },
    "collision": {
        "family": "rigid_dynamics",
        "expected_pattern": "contact should precede post-collision motion or rebound",
    },
    "elastic_motion": {
        "family": "rigid_dynamics",
        "expected_pattern": "bounce or rebound should visibly return energy without becoming unstable",
    },
    "damped_oscillation": {
        "family": "rigid_dynamics",
        "expected_pattern": "oscillation amplitude should reduce over time",
    },
    "rigid_body_motion": {
        "family": "rigid_dynamics",
        "expected_pattern": "the moving object should preserve rigid geometry and silhouette",
    },
    "deformation": {
        "family": "deformation",
        "expected_pattern": "shape should change coherently under force or contact",
    },
    "size_changing": {
        "family": "deformation",
        "expected_pattern": "size should expand or contract coherently rather than pop",
    },
    "liquid_motion": {
        "family": "fluid",
        "expected_pattern": "liquid should flow, splash, or pour with coherent fluid continuity",
    },
    "smoke_effects": {
        "family": "fluid",
        "expected_pattern": "smoke, mist, or steam should drift as a coherent plume",
    },
    "combustion": {
        "family": "fluid",
        "expected_pattern": "flames and heat-driven gaseous motion should evolve plausibly",
    },
    "melting": {
        "family": "phase_change",
        "expected_pattern": "solid material should soften or transition toward liquid state over time",
    },
    "solidification": {
        "family": "phase_change",
        "expected_pattern": "material should become more solid or frozen over time",
    },
    "vaporization": {
        "family": "phase_change",
        "expected_pattern": "liquid should transition toward vapor or steam",
    },
    "liquefaction": {
        "family": "phase_change",
        "expected_pattern": "material or vapor should transition toward a liquid-like state",
    },
    "reflection": {
        "family": "reflection",
        "expected_pattern": "the reflection should stay coupled to the scene and viewing geometry",
    },
    "refraction": {
        "family": "refraction",
        "expected_pattern": "transparent media should distort background content plausibly",
    },
    "scattering": {
        "family": "scattering",
        "expected_pattern": "light should interact with haze, fog, mist, or volume in a coherent way",
    },
    "directional_lighting": {
        "family": "lighting",
        "expected_pattern": "beams, highlights, and shadows should align with a directional source",
    },
    "point_lighting": {
        "family": "lighting",
        "expected_pattern": "falloff and highlights should align with localized point-like light sources",
    },
    "colored_lighting_bloom": {
        "family": "lighting",
        "expected_pattern": "colored glow and bloom should remain stable around light sources",
    },
}


def known_physics_dimensions() -> set[str]:
    return set(_DIMENSION_OVERRIDES)


def build_judge_card(
    *,
    group: str,
    dimension: str,
    caption: str,
    label: str | None = None,
) -> dict[str, Any]:
    normalized_dimension = str(dimension).strip().lower()
    override = _DIMENSION_OVERRIDES.get(normalized_dimension, {})
    family = str(override.get("family", "kinematics"))
    base = deepcopy(_FAMILY_DEFAULTS.get(family, _FAMILY_DEFAULTS["kinematics"]))
    base.update({key: value for key, value in override.items() if key != "family"})
    base["group"] = str(group).strip()
    base["dimension"] = normalized_dimension
    base["caption"] = str(caption or "").strip()
    if label:
        base["reference_label"] = str(label).strip()
    return base


__all__ = ["build_judge_card", "known_physics_dimensions"]
