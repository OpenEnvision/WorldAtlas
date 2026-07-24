"""Generation protocol specifications shared between benchmark and model adapters."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Sequence

from worldarena.benchmark.annotations import (
    CAMERA_MOTION_PROMPTS,
    GENERIC_OBJECT_TOKENS,
    STRUCTURED_OBJECT_KEYS,
    _clean_text,
    _dedupe_phrases,
    _derive_object_prompts_from_text,
    _flatten_strings,
)


STATIC_SCENE_LABELS = {
    "aquatic_landscape": "Aquatic Landscape",
    "city": "City Street",
    "dining_spaces": "Dining Space",
    "living_spaces": "Living Space",
    "passageways": "Passageway",
    "public_spaces": "Public Space",
    "suburb": "Suburban Street",
    "terrestrial_landscape": "Terrestrial Landscape",
    "verdant_landscape": "Verdant Landscape",
    "workspaces": "Workspace",
}
MOTION_OBJECT_KEYWORDS = {
    "articulated": {
        "animal",
        "animals",
        "bird",
        "birds",
        "body",
        "cat",
        "cats",
        "creature",
        "creatures",
        "dog",
        "dogs",
        "face",
        "fish",
        "hand",
        "horse",
        "horses",
        "human",
        "insect",
        "insects",
        "person",
        "people",
    },
    "deformable": {
        "banner",
        "banners",
        "branch",
        "branches",
        "cloth",
        "fabric",
        "flag",
        "flags",
        "foliage",
        "grass",
        "hair",
        "leaves",
        "plant",
        "plants",
        "tree",
        "trees",
    },
    "fluid": {
        "cloud",
        "clouds",
        "fire",
        "flame",
        "flames",
        "foam",
        "mist",
        "ocean",
        "rain",
        "river",
        "sea",
        "smoke",
        "spray",
        "surf",
        "water",
        "waterfall",
        "waterfalls",
        "wave",
        "waves",
    },
    "multi_motion": {
        "animal",
        "animals",
        "bird",
        "birds",
        "car",
        "cars",
        "dog",
        "dogs",
        "fish",
        "flag",
        "flags",
        "person",
        "people",
        "smoke",
        "vehicle",
        "vehicles",
        "water",
        "wave",
        "waves",
    },
    "rigid": {
        "airplane",
        "ball",
        "boat",
        "bus",
        "car",
        "cars",
        "drone",
        "plane",
        "train",
        "tram",
        "truck",
        "vehicle",
        "vehicles",
    },
}
SCENERY_OBJECT_KEYWORDS = {
    "background",
    "beach",
    "building",
    "buildings",
    "city",
    "cityscape",
    "coast",
    "coastline",
    "horizon",
    "landscape",
    "mountain",
    "mountains",
    "road",
    "rocks",
    "shore",
    "sky",
    "street",
    "village",
}
CAMERA_LABEL_TO_TOKEN = {
    "dolly in": "push_in",
    "dolly out": "pull_out",
    "zoom in": "push_in",
    "zoom out": "pull_out",
    "move left": "move_left",
    "move right": "move_right",
    "orbit left": "orbit_left",
    "orbit right": "orbit_right",
    "pan left": "pan_left",
    "pan right": "pan_right",
    "truck left": "move_left",
    "truck right": "move_right",
    "tilt up": "tilt_up",
    "tilt down": "tilt_down",
    "roll cw": "roll_cw",
    "roll ccw": "roll_ccw",
    "pedestal up": "pedestal_up",
    "pedestal down": "pedestal_down",
    "stay": "fixed",
}
WEAK_SCENE_LABELS = {
    "dynamic",
    "dynamic scene",
    "indoor",
    "motion",
    "outdoor",
    "scene",
    "unknown",
}
GENERIC_OBJECT_END_TOKENS = {"ambience", "atmosphere", "background", "lighting", "motion", "scene", "view"}
GENERIC_OBJECT_SUBTOKENS = {"camera", "framing", "perspective", "shot", "viewer"}
GENERIC_SCENE_OBJECT_END_TOKENS = {
    "area",
    "cityscape",
    "corridor",
    "hallway",
    "intersection",
    "landscape",
    "path",
    "promenade",
    "room",
    "scene",
    "space",
    "station",
    "street",
    "studio",
}
BAD_OBJECT_TOKENS = {
    "blend",
    "captures",
    "exudes",
    "glides",
    "indoor",
    "modern",
    "observed",
    "outdoor",
    "rainy",
    "scenic",
    "serene",
    "showcases",
    "stylized",
    "sunny",
    "urban",
}
BAD_SINGLE_OBJECT_TOKENS = {
    "approach",
    "architectural",
    "bustling",
    "down",
    "indoor",
    "living",
    "modern",
    "nighttime",
    "prominent",
    "quiet",
    "rainy",
    "scenic",
    "serene",
    "showcases",
    "street",
    "stylized",
    "sunny",
    "the",
    "urban",
}
MOTION_CATEGORY_LABELS = {
    "articulated": "articulated",
    "deformable": "deformable",
    "fluid": "fluid",
    "multi_motion": "multi-motion",
    "rigid": "rigid",
}


def title_case_slug(value: str | None) -> str:
    if not value:
        return "Unknown"
    return value.replace("_", " ").replace("-", " ").title()


def _clean_sentence(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _human_join(values: Sequence[str]) -> str:
    items = [_clean_text(value).lower() for value in values if _clean_text(value)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _scene_sentence(scene_label: str) -> str:
    label = _clean_text(scene_label).lower()
    if not label:
        return "A scene"
    words = label.split()
    if not words:
        return "A scene"
    if words[-1].endswith("s"):
        return label.capitalize()
    article = "an" if words[0][0] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}".capitalize()


def _content_text(scene_label: str, objects: Sequence[str]) -> str:
    scene_text = _clean_text(scene_label).strip(" .,;:-")
    scene_key = scene_text.lower()
    parts = [scene_text]
    seen = {scene_key}
    for obj in objects[:3]:
        value = _clean_text(obj).strip(" .,;:-")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(value)
    return ", ".join(part for part in parts if part)


def _scene_prompt(scene_label: str, objects: Sequence[str]) -> str:
    prefix = _scene_sentence(scene_label)
    scene_key = _clean_text(scene_label).lower().strip(" .,;:-")
    prompt_objects = [obj for obj in objects[:3] if _clean_text(obj).lower().strip(" .,;:-") != scene_key]
    detail_text = _human_join(prompt_objects)
    if detail_text:
        return f"{prefix} with {detail_text}."
    return f"{prefix}."


def _normalize_object_phrase(value: str) -> str | None:
    text = _clean_text(value).lower().strip(" .,;:-")
    if not text or text in GENERIC_OBJECT_TOKENS:
        return None
    text = re.sub(r"^(?:a|an|the|its|their|some|several|many|multiple)\s+", "", text)
    text = re.split(r"\b(?:that|which|while|where|with)\b", text, maxsplit=1)[0].strip(" .,;:-")
    if " like " in text:
        text = text.split(" like ", 1)[1].strip(" .,;:-")
    tokens = [piece for piece in re.split(r"[\s/-]+", text) if piece]
    if not tokens:
        return None
    if set(tokens) & GENERIC_OBJECT_SUBTOKENS:
        return None
    if set(tokens) & BAD_OBJECT_TOKENS:
        return None
    if len(tokens) == 1 and tokens[0] in BAD_SINGLE_OBJECT_TOKENS:
        return None
    if tokens[-1] in GENERIC_OBJECT_END_TOKENS:
        return None
    if len(tokens) == 1 and tokens[-1] in GENERIC_SCENE_OBJECT_END_TOKENS:
        return None
    if len(tokens) > 6:
        return None
    return text


def _heuristic_object_phrases(text: str) -> list[str]:
    lowered = _clean_text(text).lower()
    if not lowered:
        return []

    candidates: list[str] = []
    segment_patterns = (
        re.compile(r"\bwith\s+([^.;:]+)"),
        re.compile(r"\bfeaturing\s+([^.;:]+)"),
        re.compile(r"\bfeatures\s+([^.;:]+)"),
        re.compile(r"\bincluding\s+([^.;:]+)"),
        re.compile(r"\bincludes\s+([^.;:]+)"),
        re.compile(r"\blike\s+([^.;:]+)"),
    )
    for pattern in segment_patterns:
        for match in pattern.finditer(lowered):
            segment = match.group(1)
            for piece in re.split(r",| and ", segment):
                cleaned = piece.strip(" .,;:-")
                if cleaned:
                    candidates.append(cleaned)

    for match in re.finditer(r"\b(?:a|an|the)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3})", lowered):
        candidates.append(match.group(1))

    return candidates


def _derive_objects(caption_payload: dict[str, Any], *, limit: int = 8) -> list[str]:
    explicit_values: list[str] = []
    for key in STRUCTURED_OBJECT_KEYS:
        explicit_values.extend(_flatten_strings(caption_payload.get(key)))
    category_tags = caption_payload.get("CategoryTags")
    if isinstance(category_tags, dict):
        explicit_values.extend(_flatten_strings(category_tags.get("objects")))
        explicit_values.extend(_flatten_strings(category_tags.get("entities")))

    text = " ".join(
        part
        for part in (
            _clean_text(caption_payload.get("SceneSummary")),
            _clean_text(caption_payload.get("SceneDescription")),
        )
        if part
    )
    normalized: list[str] = []
    seen: set[str] = set()
    def _consume(values: Sequence[str]) -> None:
        for value in values:
            phrase = _normalize_object_phrase(value)
            if phrase is None or phrase in seen:
                continue
            seen.add(phrase)
            normalized.append(phrase)
            if len(normalized) >= limit:
                break

    _consume([value for value in explicit_values if isinstance(value, str)])
    if normalized:
        return normalized[:limit]
    if len(normalized) < limit:
        _consume(_heuristic_object_phrases(text))
    if len(normalized) < min(limit, 3):
        _consume(_derive_object_prompts_from_text(text, limit=max(limit * 2, 8)))
    return normalized


def _sorted_instruction_values(instruction_payload: dict[str, Any]) -> list[str]:
    ordered: list[str] = []

    def _sort_key(item: tuple[str, Any]) -> tuple[int, int, str]:
        key = str(item[0])
        match = re.match(r"(\d+)->(\d+)", key)
        if match:
            return (int(match.group(1)), int(match.group(2)), key)
        return (10**9, 10**9, key)

    for _, values in sorted(instruction_payload.items(), key=_sort_key):
        ordered.extend(_flatten_strings(values))
    return ordered


def normalize_camera_path(instruction_payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw_values = _sorted_instruction_values(instruction_payload)
    tokens: list[str] = []
    phrases: list[str] = []
    for raw_value in raw_values:
        cleaned = _clean_text(raw_value)
        if not cleaned:
            continue
        token = CAMERA_LABEL_TO_TOKEN.get(cleaned.lower(), cleaned.lower().replace("-", "_").replace(" ", "_"))
        phrase = CAMERA_MOTION_PROMPTS.get(token, cleaned.lower())
        tokens.append(token)
        phrases.append(phrase)
    if not tokens:
        return ["fixed"], ["fixed"]
    return tokens, phrases


def _scene_label_from_caption(
    caption_payload: dict[str, Any],
    *,
    motion_regime: str,
    environment: str | None = None,
    scene: str | None = None,
    motion_category: str | None = None,
) -> str:
    if motion_regime == "static" and scene:
        return STATIC_SCENE_LABELS.get(scene, title_case_slug(scene))

    category_tags = caption_payload.get("CategoryTags")
    scene_type = category_tags.get("sceneType") if isinstance(category_tags, dict) else {}
    candidates = []
    if isinstance(scene_type, dict):
        candidates.extend(
            _clean_text(scene_type.get(key))
            for key in ("second", "first")
        )
    for candidate in candidates:
        if not candidate:
            continue
        label = title_case_slug(candidate).strip()
        lowered = label.lower()
        if lowered in WEAK_SCENE_LABELS:
            continue
        motion_label = MOTION_CATEGORY_LABELS.get(motion_category or "", "")
        if motion_label and lowered == motion_label.lower():
            continue
        if len(label.split()) == 1:
            return f"{label} Scene"
        return label

    if motion_regime == "static":
        if scene:
            return STATIC_SCENE_LABELS.get(scene, title_case_slug(scene))
        if environment == "indoor":
            return "Indoor Scene"
        if environment == "outdoor":
            return "Outdoor Scene"
        return "Static Scene"

    if motion_category:
        return f"{title_case_slug(motion_category)} Scene"
    return "Dynamic Scene"


def _augment_caption_payload(caption_payload: dict[str, Any], objects: list[str]) -> dict[str, Any]:
    updated = deepcopy(caption_payload)
    updated["objects"] = list(objects)
    category_tags = updated.get("CategoryTags")
    if not isinstance(category_tags, dict):
        category_tags = {}
        updated["CategoryTags"] = category_tags
    category_tags["objects"] = list(objects)
    return updated


def _next_scene_objects(objects: list[str], *, step_index: int, layout: str) -> list[str]:
    if not objects:
        return []
    if len(objects) <= 3:
        return list(objects)
    if layout in {"push_in", "orbit_left", "orbit_right", "tilt_up", "tilt_down", "roll_cw", "roll_ccw"}:
        return list(objects[:3])
    start = step_index % len(objects)
    rotated = objects[start:] + objects[:start]
    return list(rotated[:3])


def _dynamic_prompt(objects: list[str], motion_category: str | None) -> str:
    motion_label = MOTION_CATEGORY_LABELS.get(motion_category or "", title_case_slug(motion_category)).strip("- ")
    subject = _dynamic_subject(objects, motion_category)
    subject_text = subject if subject.startswith(("a ", "an ", "the ")) else f"the {subject}"
    return (
        f"There is clear {motion_label} motion in {subject_text} while the surrounding scene remains coherent."
        if motion_label
        else f"There is clear motion in {subject_text} while the surrounding scene remains coherent."
    )


def _dynamic_subject(objects: Sequence[str], motion_category: str | None) -> str:
    if not objects:
        return "main subject"

    keyword_hits = MOTION_OBJECT_KEYWORDS.get(motion_category or "", set())
    best_object = objects[0]
    best_score = -10**9
    for index, obj in enumerate(objects):
        text = _clean_text(obj).lower()
        tokens = set(re.split(r"[\s/-]+", text))
        score = 0
        score -= index
        score += 4 * len(tokens & keyword_hits)
        if motion_category == "fluid" and "water" in tokens and "waves" in text:
            score += 3
        if tokens & SCENERY_OBJECT_KEYWORDS:
            score -= 2
        if text in {"rocky coastline", "city street", "mountain landscape"}:
            score -= 3
        if score > best_score:
            best_object = obj
            best_score = score
    return best_object


def build_video_generation_spec(
    caption_payload: dict[str, Any],
    instruction_payload: dict[str, Any],
    *,
    motion_regime: str,
    environment: str | None = None,
    scene: str | None = None,
    motion_category: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = _derive_objects(caption_payload, limit=8)
    if not objects:
        scene_label = _scene_label_from_caption(
            caption_payload,
            motion_regime=motion_regime,
            environment=environment,
            scene=scene,
            motion_category=motion_category,
        )
        fallback = _clean_text(scene_label).lower()
        if fallback:
            objects = [fallback]
    caption_payload = _augment_caption_payload(caption_payload, objects)

    scene_label = _scene_label_from_caption(
        caption_payload,
        motion_regime=motion_regime,
        environment=environment,
        scene=scene,
        motion_category=motion_category,
    )
    current_objects = list(objects[:4])
    current_content = _content_text(scene_label, current_objects)
    current_prompt = _clean_sentence(caption_payload.get("SceneSummary")) or _scene_prompt(scene_label, current_objects)
    camera_path, camera_phrases = normalize_camera_path(instruction_payload)

    if motion_regime == "dynamic":
        moving_objects = list(objects[:3]) or ["main subject"]
        prompt = _dynamic_prompt(moving_objects, motion_category)
        return caption_payload, {
            "mode": "dynamic",
            "camera_path": list(camera_path),
            "objects": moving_objects,
            "current_objects": current_objects,
            "content": current_content,
            "current_prompt": current_prompt,
            "prompt": prompt,
            "content_list": [current_content],
            "prompt_list": [current_prompt],
            "scene_objects": [current_objects],
            "segment_prompts": [f"Camera {camera_phrases[0]}. {prompt}"],
        }

    content_list = [current_content]
    prompt_list = [current_prompt]
    scene_objects = [current_objects]
    segment_prompts: list[str] = []
    for step_index, (camera_token, camera_phrase) in enumerate(zip(camera_path, camera_phrases), start=1):
        next_objects = _next_scene_objects(objects, step_index=step_index, layout=camera_token)
        next_content = _content_text(scene_label, next_objects)
        next_prompt = _scene_prompt(scene_label, next_objects)
        content_list.append(next_content)
        prompt_list.append(next_prompt)
        scene_objects.append(next_objects)
        segment_prompts.append(f"Camera {camera_phrase}. {next_prompt}")

    return caption_payload, {
        "mode": "static",
        "camera_path": list(camera_path),
        "objects": current_objects,
        "current_objects": current_objects,
        "content": current_content,
        "current_prompt": current_prompt,
        "content_list": content_list,
        "prompt_list": prompt_list,
        "scene_objects": scene_objects,
        "segment_prompts": segment_prompts,
    }


__all__ = [
    "build_video_generation_spec",
    "normalize_camera_path",
]
