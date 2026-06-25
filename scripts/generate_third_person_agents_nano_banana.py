from __future__ import annotations

import argparse
import base64
import concurrent.futures
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any


DEFAULT_DATA_ROOT = Path(
    "/mnt/cpfs/yangboxue/visual_generation/juanxi/data/OpenEnvision/worldarena/selected_static_1000"
)
DEFAULT_PROMPT_FILE = DEFAULT_DATA_ROOT / "WorldArena_third_person_agents_total_prompt_file.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT.parent / "selected_static_1000_third_person_agents_nano_banana"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL = "gemini-3.1-flash-image"
DEFAULT_VALIDATION_MODEL = "gemini-3-pro-preview"
SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_HARASSMENT",
)
BACK_VIEW_REQUIREMENT = (
    "Critical orientation requirement: the inserted agent must be back-facing, "
    "with its back toward the camera/viewer. This applies to both humans and animals. "
    "For humans, show the back of the head, back, shoulders, and rear-facing body only; "
    "do not show the face, eyes, front torso, profile, or three-quarter face. "
    "For animals, show the rear/back-facing body and head oriented away from the camera; "
    "do not show the face, eyes, snout, muzzle, profile, or three-quarter face. "
    "If the subject or motion text mentions looking back, looking to the side, or turning, "
    "reinterpret it as a body/movement pose while the head and body remain oriented away from the camera. "
    "Do not make an over-the-shoulder glance."
)
BACK_VIEW_NEGATIVE_PROMPT = (
    "no frontal view, no face, no visible eyes, no eye contact, no looking at the camera, "
    "no side profile, no profile head, no three-quarter face, no visible snout, no visible muzzle, "
    "no front torso, no over-the-shoulder glance, no head turned back, no head turned sideways"
)
GAME_LIKE_COMPOSITION_REQUIREMENT = (
    "Game-like third-person composition requirement: place the agent on a plausible walkable or navigable "
    "surface near the exact horizontal center and lower-middle foreground of the image, like a playable "
    "third-person game character seen from behind. Prefer the agent's torso or body center at the image "
    "centerline, roughly within the central 40-60% of the image width, and aligned with the scene's central "
    "walkable path when one exists. The agent should appear closer to the camera and visually prominent, "
    "not tiny, distant, peripheral, or hidden. Keep the full body visible with head, feet, and limbs uncropped. "
    "As a target, make a human agent roughly 28-45% of the image height and an animal agent roughly 20-35% "
    "of the image height, adjusted to the scene perspective. Do not place the agent at the left/right side, "
    "extreme edge, deep background, or behind major occluders. Do not replace or warp the background camera "
    "view; achieve the closer third-person feel through perspective-consistent scale and foreground placement."
)
GAME_LIKE_COMPOSITION_NEGATIVE_PROMPT = (
    "no tiny agent, no distant background agent, no edge placement, no corner placement, no hidden agent, "
    "no cropped full body, no foreground cut-off, no agent behind major occluders, no left-side placement, "
    "no right-side placement, no off-center avatar"
)


@dataclass(frozen=True)
class GenerationTask:
    index: int
    sample_id: str
    source_path: str
    source_relative_path: str
    output_path: str
    prompt_id: str
    subject_type: str
    category: str
    subject: str
    motion: str
    prompt: str
    negative_prompt: str
    control_tags: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_unit(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _prompt_records(prompt_file: Path) -> list[dict[str, Any]]:
    payload = _read_json(prompt_file)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"prompt file has no records list: {prompt_file}")
    normalized: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"record_{len(normalized):04d}"),
                "subject_type": str(item.get("subject_type") or "human"),
                "category": str(item.get("category") or "general"),
                "subject": str(item.get("subject") or ""),
                "motion": str(item.get("motion") or ""),
                "prompt": prompt,
                "negative_prompt": str(item.get("negative_prompt") or ""),
                "control_tags": list(item.get("control_tags") or []),
            }
        )
    if not normalized:
        raise ValueError(f"prompt file has no usable records: {prompt_file}")
    return normalized


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith(".")
    )


def _preferred_subject_type(relative_path: Path, *, seed: int) -> str:
    parts = [part.lower() for part in relative_path.parts]
    environment = parts[1] if len(parts) > 1 else ""
    scene = parts[2] if len(parts) > 2 else ""
    human_probability = 0.65
    if environment == "indoor":
        human_probability = 0.82
    elif scene in {"city", "suburb"}:
        human_probability = 0.74
    elif scene in {"verdant_landscape", "terrestrial_landscape"}:
        human_probability = 0.45
    elif scene == "aquatic_landscape":
        human_probability = 0.40
    return "human" if _hash_unit(relative_path.as_posix(), seed) < human_probability else "animal"


def _preferred_categories(relative_path: Path, subject_type: str) -> list[str]:
    parts = [part.lower() for part in relative_path.parts]
    style = parts[0] if len(parts) > 0 else ""
    environment = parts[1] if len(parts) > 1 else ""
    scene = parts[2] if len(parts) > 2 else ""

    if subject_type == "human":
        if scene in {"workspaces", "public_spaces", "city"}:
            categories = ["urban_everyday", "professional", "body_age_accessibility"]
        elif scene in {"dining_spaces", "living_spaces", "passageways"}:
            categories = ["urban_everyday", "body_age_accessibility", "style_diverse"]
        elif scene in {"verdant_landscape", "terrestrial_landscape", "aquatic_landscape"}:
            categories = ["sports_high_motion", "style_diverse", "urban_everyday"]
        else:
            categories = ["urban_everyday", "style_diverse", "professional"]
        if style == "stylized":
            categories = ["style_diverse", *categories]
        return categories

    if environment == "indoor":
        return ["pets", "birds"]
    if scene == "aquatic_landscape":
        return ["aquatic_amphibious", "birds", "wild_land", "pets"]
    if scene in {"verdant_landscape", "terrestrial_landscape"}:
        return ["wild_land", "farm_animals", "birds", "pets"]
    if scene in {"city", "suburb"}:
        return ["pets", "birds", "wild_land"]
    return ["pets", "wild_land", "birds"]


def _build_prompt_pools(records: list[dict[str, Any]], *, seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rng = random.Random(seed)
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {"human": {}, "animal": {}}
    for record in records:
        subject_type = record["subject_type"]
        pools.setdefault(subject_type, {}).setdefault(record["category"], []).append(record)
    for category_map in pools.values():
        for items in category_map.values():
            rng.shuffle(items)
    return pools


def _select_record(
    *,
    relative_path: Path,
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    counters: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    subject_type = _preferred_subject_type(relative_path, seed=seed)
    category_map = pools.get(subject_type) or {}
    preferred = _preferred_categories(relative_path, subject_type)
    candidates: list[dict[str, Any]] = []
    for category in preferred:
        candidates.extend(category_map.get(category, []))
    if not candidates:
        for items in category_map.values():
            candidates.extend(items)
    if not candidates:
        raise ValueError(f"no prompt records available for subject_type={subject_type}")

    key = f"{subject_type}:{','.join(preferred)}"
    index = counters.get(key, 0)
    counters[key] = index + 1
    return candidates[index % len(candidates)]


def _back_view_safe_prompt(record_prompt: str) -> str:
    replacements = [
        (
            r"\bturning (his|her|their|its) upper body as if looking back while walking\b",
            "walking away from the camera with a natural rear-view stride",
        ),
        (
            r"\bturning gently while stepping forward\b",
            "stepping away from the camera in a gentle rear-facing motion",
        ),
        (
            r"\bstopping and looking to the (right|left|side)\b",
            "stopping in a rear-facing stance with the head still facing away from the camera",
        ),
        (
            r"\bslightly looking to the (right|left|side)\b",
            "keeping the head facing away from the camera",
        ),
        (
            r"\blooking (back|to the side|to the right|to the left|right|left|sideways)\b",
            "facing away from the camera",
        ),
    ]
    safe_prompt = record_prompt
    for pattern, replacement in replacements:
        safe_prompt = re.sub(pattern, replacement, safe_prompt, flags=re.IGNORECASE)
    return safe_prompt


def _edit_instruction(record_prompt: str, negative_prompt: str) -> str:
    parts = [
        "Edit the provided image. Use it as the fixed background.",
        _back_view_safe_prompt(record_prompt),
        BACK_VIEW_REQUIREMENT,
        GAME_LIKE_COMPOSITION_REQUIREMENT,
        "Output only one edited image. Keep the same aspect ratio, camera view, scene layout, style, and background details.",
        "The inserted agent must be a single controllable third-person subject with plausible movement affordance.",
    ]
    avoid = ", ".join(
        part
        for part in [
            negative_prompt,
            BACK_VIEW_NEGATIVE_PROMPT,
            GAME_LIKE_COMPOSITION_NEGATIVE_PROMPT,
        ]
        if part
    )
    if avoid:
        parts.append(f"Avoid: {avoid}.")
    return "\n".join(parts)


def build_tasks(
    *,
    image_root: Path,
    output_root: Path,
    prompt_file: Path,
    seed: int,
    limit: int | None,
    start_index: int,
) -> list[GenerationTask]:
    records = _prompt_records(prompt_file)
    pools = _build_prompt_pools(records, seed=seed)
    counters: dict[str, int] = {}
    images = _image_paths(image_root)
    if start_index:
        images = images[start_index:]
    if limit is not None:
        images = images[:limit]

    tasks: list[GenerationTask] = []
    for index, source_path in enumerate(images, start=start_index):
        relative_path = source_path.relative_to(image_root)
        record = _select_record(
            relative_path=relative_path,
            pools=pools,
            counters=counters,
            seed=seed,
        )
        output_path = output_root / relative_path.with_suffix(".png")
        sample_id = relative_path.with_suffix("").as_posix().replace("/", "__")
        tasks.append(
            GenerationTask(
                index=index,
                sample_id=sample_id,
                source_path=str(source_path),
                source_relative_path=relative_path.as_posix(),
                output_path=str(output_path),
                prompt_id=record["id"],
                subject_type=record["subject_type"],
                category=record["category"],
                subject=record["subject"],
                motion=record["motion"],
                prompt=_edit_instruction(record["prompt"], record["negative_prompt"]),
                negative_prompt=record["negative_prompt"],
                control_tags=[
                    *list(record["control_tags"]),
                    "back_view",
                    "facing_away_from_camera",
                    "game_like_third_person_view",
                    "center_foreground",
                    "close_camera",
                ],
            )
        )
    return tasks


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if suffixes & {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ".png" in suffixes:
        return "image/png"
    if ".webp" in suffixes:
        return "image/webp"
    return "application/octet-stream"


def _load_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(
            "google-genai is not installed in this Python environment. "
            "Install it with `python -m pip install google-genai`."
        ) from exc
    return genai, types


def _api_key(env_names: list[str]) -> str | None:
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _make_client(args: argparse.Namespace) -> Any:
    genai, types = _load_genai()
    api_key = _api_key(args.api_key_env)
    kwargs: dict[str, Any] = {}
    timeout_ms = max(int(float(args.request_timeout_seconds) * 1000), 1)
    kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
    if args.vertexai:
        kwargs["vertexai"] = True
        if args.project:
            kwargs["project"] = args.project
        if args.location:
            kwargs["location"] = args.location
    if api_key:
        kwargs["api_key"] = api_key
    return genai.Client(**kwargs)


def _generation_config(args: argparse.Namespace) -> Any:
    _, types = _load_genai()
    safety_settings = None
    if args.safety_off:
        safety_settings = [
            types.SafetySetting(category=category, threshold="OFF")
            for category in SAFETY_CATEGORIES
        ]
    tools = None
    if args.google_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]

    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "response_modalities": ["TEXT", "IMAGE"],
    }
    if safety_settings is not None:
        kwargs["safety_settings"] = safety_settings
    if tools is not None:
        kwargs["tools"] = tools
    if args.thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=args.thinking_level)
    return types.GenerateContentConfig(**kwargs)


def _content_parts(task: GenerationTask) -> list[Any]:
    _, types = _load_genai()
    source_path = Path(task.source_path)
    image_bytes = source_path.read_bytes()
    return [
        task.prompt,
        types.Part.from_bytes(data=image_bytes, mime_type=_mime_type(source_path)),
    ]


def _save_response_image(response: Any, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_parts: list[str] = []
    inline_count = 0

    parts = getattr(response, "parts", None)
    if parts is None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None)
    if not parts:
        raise RuntimeError("Gemini response contained no parts")

    for part in parts:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
            continue
        inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline_data is None:
            continue
        inline_count += 1
        data = getattr(inline_data, "data", None)
        if data is None:
            continue
        if isinstance(data, str):
            data = base64.b64decode(data)
        output_path.write_bytes(bytes(data))
        return {
            "text_parts": text_parts,
            "inline_image_count": inline_count,
            "image_size_bytes": output_path.stat().st_size,
        }

    for part in parts:
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = as_image()
            image.save(output_path)
            return {
                "text_parts": text_parts,
                "inline_image_count": inline_count,
                "image_size_bytes": output_path.stat().st_size,
            }

    raise RuntimeError("Gemini response contained no image part")


def _temporary_output_path(output_path: Path) -> Path:
    suffix = f".tmp.{os.getpid()}.{threading.get_ident()}"
    return output_path.with_name(f".{output_path.name}{suffix}")


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)

    parts = getattr(response, "parts", None)
    if parts is None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None)
    if not parts:
        return ""

    text_parts: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text:
            text_parts.append(str(part_text))
    return "\n".join(text_parts)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _validation_config() -> Any:
    _, types = _load_genai()
    return types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.1,
        max_output_tokens=512,
    )


def _validate_back_view(
    client: Any,
    task: GenerationTask,
    edited_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.validate_back_view:
        return {
            "accepted": True,
            "model": None,
            "reason": "back-view validation disabled",
        }

    _, types = _load_genai()
    source_path = Path(task.source_path)
    prompt = (
        "You are validating an image-editing dataset sample.\n"
        "Image 1 is the original background. Image 2 is the edited output.\n"
        "Decide whether the edited output added exactly one controllable third-person human or animal agent, "
        "and whether that newly added agent is strictly back-facing with its back toward the camera/viewer.\n"
        "Accept only if the newly added agent's face is not visible. For humans, reject visible face, eyes, "
        "front torso, side profile, or three-quarter face. For animals, reject visible face, eyes, snout, muzzle, "
        "side profile, or three-quarter face. Also reject if there are multiple newly added agents, no clear "
        "new agent, cropped full body, or a poster/toy/statue-like insertion. Also reject if the new agent is "
        "clearly tiny, very distant, hidden, at the extreme image edge, noticeably off to the left/right side, "
        "or not readable as a close third-person game-like controllable avatar centered in the lower-middle "
        "foreground.\n"
        "Return JSON only in this schema: "
        "{\"accepted\": true|false, \"reason\": \"short reason\", \"orientation\": \"back|front|side|unclear\"}."
    )
    response = client.models.generate_content(
        model=args.validation_model,
        contents=[
            prompt,
            types.Part.from_bytes(data=source_path.read_bytes(), mime_type=_mime_type(source_path)),
            types.Part.from_bytes(data=edited_path.read_bytes(), mime_type=_mime_type(edited_path)),
        ],
        config=_validation_config(),
    )
    text = _response_text(response).strip()
    payload = _parse_json_object(text)
    if payload is None:
        return {
            "accepted": False,
            "model": args.validation_model,
            "reason": "validation response was not parseable JSON",
            "raw_text": text[:1000],
        }

    return {
        "accepted": payload.get("accepted") is True,
        "model": args.validation_model,
        "reason": str(payload.get("reason") or ""),
        "orientation": str(payload.get("orientation") or "unclear"),
        "raw_text": text[:1000],
    }


def _generate_one(
    task: GenerationTask,
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    errors_path: Path,
    lock: threading.Lock,
) -> str:
    output_path = Path(task.output_path)
    if output_path.exists() and not args.overwrite:
        return "skipped_existing"

    client = _make_client(args)
    config = _generation_config(args)
    last_error: Exception | None = None
    last_validation: dict[str, Any] | None = None
    validation_attempts = 1
    if args.validate_back_view:
        validation_attempts += max(int(args.back_view_validation_retries), 0)

    for validation_attempt in range(validation_attempts):
        tmp_output_path = _temporary_output_path(output_path)
        try:
            for attempt in range(args.max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=args.model,
                        contents=_content_parts(task),
                        config=config,
                    )
                    result_details = _save_response_image(response, tmp_output_path)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < args.max_retries:
                        time.sleep(args.retry_backoff * (2**attempt))
            else:
                break

            validation = _validate_back_view(client, task, tmp_output_path, args=args)
            last_validation = validation
            if not validation.get("accepted"):
                if validation_attempt + 1 < validation_attempts:
                    tmp_output_path.unlink(missing_ok=True)
                    continue
                break

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_output_path.replace(output_path)
            result_details["back_view_validation"] = validation
            result_details["back_view_validation_attempt"] = validation_attempt + 1
            _append_jsonl(
                manifest_path,
                {
                    **asdict(task),
                    "status": "generated",
                    "model": args.model,
                    "generated_at": _utc_now(),
                    "details": result_details,
                },
                lock,
            )
            return "generated"
        except Exception as exc:
            last_error = exc
            break
        finally:
            tmp_output_path.unlink(missing_ok=True)

    _append_jsonl(
        errors_path,
        {
            **asdict(task),
            "status": "failed",
            "model": args.model,
            "failed_at": _utc_now(),
            "error": str(last_error),
            "back_view_validation": last_validation,
        },
        lock,
    )
    return "failed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate third-person controllable-agent image edits for selected WorldArena static images."
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--validation-model", default=DEFAULT_VALIDATION_MODEL)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--errors-path", type=Path, default=None)
    parser.add_argument("--api-key-env", nargs="+", default=["GOOGLE_CLOUD_API_KEY", "GEMINI_API_KEY"])
    parser.add_argument("--vertexai", action="store_true")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_CLOUD_REGION"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--thinking-level", default=None)
    parser.add_argument("--google-search", action="store_true")
    parser.add_argument("--safety-off", action="store_true")
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--validate-back-view", dest="validate_back_view", action="store_true", default=True)
    parser.add_argument("--no-validate-back-view", dest="validate_back_view", action="store_false")
    parser.add_argument("--back-view-validation-retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()
    prompt_file = args.prompt_file.resolve()
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path
        else output_root / "third_person_agents_generation_manifest.jsonl"
    )
    errors_path = (
        args.errors_path.resolve()
        if args.errors_path
        else output_root / "third_person_agents_generation_errors.jsonl"
    )
    plan_path = (
        args.plan_path.resolve()
        if args.plan_path
        else output_root / "third_person_agents_generation_plan.jsonl"
    )

    tasks = build_tasks(
        image_root=image_root,
        output_root=output_root,
        prompt_file=prompt_file,
        seed=args.seed,
        limit=args.limit,
        start_index=args.start_index,
    )
    _write_jsonl(plan_path, [asdict(task) for task in tasks])

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "tasks": len(tasks),
                    "plan_path": str(plan_path),
                    "output_root": str(output_root),
                    "model": args.model,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if not _api_key(args.api_key_env) and not args.vertexai:
        print(
            "No API key found. Set GOOGLE_CLOUD_API_KEY or GEMINI_API_KEY, or pass --vertexai with ADC/project config.",
            file=sys.stderr,
        )
        return 2

    lock = threading.Lock()
    counts = {"generated": 0, "skipped_existing": 0, "failed": 0}
    workers = max(int(args.workers), 1)
    if workers == 1:
        for task in tasks:
            status = _generate_one(
                task,
                args=args,
                manifest_path=manifest_path,
                errors_path=errors_path,
                lock=lock,
            )
            counts[status] = counts.get(status, 0) + 1
            print(f"[{counts['generated'] + counts['skipped_existing'] + counts['failed']}/{len(tasks)}] {status}: {task.source_relative_path}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _generate_one,
                    task,
                    args=args,
                    manifest_path=manifest_path,
                    errors_path=errors_path,
                    lock=lock,
                ): task
                for task in tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                status = future.result()
                counts[status] = counts.get(status, 0) + 1
                completed += 1
                print(f"[{completed}/{len(tasks)}] {status}: {task.source_relative_path}", flush=True)

    print(
        json.dumps(
            {
                "status": "done",
                "tasks": len(tasks),
                **counts,
                "plan_path": str(plan_path),
                "manifest_path": str(manifest_path),
                "errors_path": str(errors_path),
                "output_root": str(output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
