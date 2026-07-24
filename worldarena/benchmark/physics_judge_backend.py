"""LLM judge backend for physics plausibility evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.physics_judge_taxonomy import build_judge_card
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.serialization import write_json


PHYSICS_JUDGE_METRICS: tuple[str, ...] = (
    "physics_judge_overall",
    "physics_judge_phenomenon",
    "physics_judge_causality",
    "physics_judge_material",
    "physics_judge_temporal",
    "physics_judge_prompt",
)
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "phenomenon": 0.25,
    "causality": 0.25,
    "material": 0.2,
    "temporal": 0.2,
    "prompt": 0.1,
}


def is_physics_judge_metric(metric_name: str) -> bool:
    return metric_name in PHYSICS_JUDGE_METRICS


def _cache_key(
    *,
    sample: BenchmarkSample,
    prediction: LoadedMedia,
    backend: str,
    runtime: dict[str, Any],
) -> str:
    stat = Path(prediction.path).stat()
    runtime_token = json.dumps(runtime, sort_keys=True, default=str)
    payload = {
        "sample_id": sample.sample_id,
        "prediction_path": str(Path(prediction.path).resolve()),
        "prediction_mtime_ns": stat.st_mtime_ns,
        "prediction_size": stat.st_size,
        "backend": backend,
        "runtime": runtime_token,
        "physics_spec": sample.physics_spec or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        return payload
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in response")


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _frame_to_jpeg_bytes(frame: np.ndarray, max_side: int, jpeg_quality: int) -> bytes:
    rgb = np.asarray(frame)
    height, width = rgb.shape[:2]
    longest_side = max(height, width)
    if max_side > 0 and longest_side > max_side:
        scale = max_side / float(longest_side)
        resized = cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not success:
        raise RuntimeError("failed to encode frame as JPEG")
    return encoded.tobytes()


def _sample_frames(media: LoadedMedia, limit: int) -> list[np.ndarray]:
    if limit <= 0:
        return []
    if len(media.frames) <= limit:
        return media.frames
    indices = np.linspace(0, len(media.frames) - 1, num=limit)
    return [media.frames[int(round(index))] for index in indices]


def _openai_content_items(
    *,
    prompt: str,
    prediction_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    max_side: int,
    jpeg_quality: int,
    detail: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if reference_frames:
        content.append({"type": "input_text", "text": "Reference frames for the target physical phenomenon."})
        for index, frame in enumerate(reference_frames, start=1):
            image_b64 = base64.b64encode(_frame_to_jpeg_bytes(frame, max_side, jpeg_quality)).decode("ascii")
            content.append({"type": "input_text", "text": f"REFERENCE_FRAME_{index}"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": detail,
                }
            )
    content.append({"type": "input_text", "text": "Generated prediction frames to judge."})
    for index, frame in enumerate(prediction_frames, start=1):
        image_b64 = base64.b64encode(_frame_to_jpeg_bytes(frame, max_side, jpeg_quality)).decode("ascii")
        content.append({"type": "input_text", "text": f"PREDICTION_FRAME_{index}"})
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
                "detail": detail,
            }
        )
    return content


def _gemini_parts(
    *,
    prompt: str,
    prediction_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    max_side: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if reference_frames:
        parts.append({"text": "Reference frames for the target physical phenomenon."})
        for index, frame in enumerate(reference_frames, start=1):
            image_b64 = base64.b64encode(_frame_to_jpeg_bytes(frame, max_side, jpeg_quality)).decode("ascii")
            parts.append({"text": f"REFERENCE_FRAME_{index}"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    parts.append({"text": "Generated prediction frames to judge."})
    for index, frame in enumerate(prediction_frames, start=1):
        image_b64 = base64.b64encode(_frame_to_jpeg_bytes(frame, max_side, jpeg_quality)).decode("ascii")
        parts.append({"text": f"PREDICTION_FRAME_{index}"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    return parts


def _read_http_json(request: urllib.request.Request, timeout_seconds: int) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("provider did not return a JSON object")
    return payload


def _openai_text_from_response(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _gemini_text_from_response(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in payload.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _load_api_key(runtime: dict[str, Any], *, default_env: str) -> str:
    raw = runtime.get("api_key")
    if raw:
        return str(raw).strip()
    env_name = str(runtime.get("api_key_env", default_env))
    import os

    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise RuntimeError(f"missing API key; set {env_name} or provide metric_runtime.api_key")


def _call_openai(
    *,
    prompt: str,
    prediction_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    api_key = _load_api_key(runtime, default_env="OPENAI_API_KEY")
    body = {
        "model": str(runtime.get("model", "gpt-4.1")),
        "input": [
            {
                "role": "user",
                "content": _openai_content_items(
                    prompt=prompt,
                    prediction_frames=prediction_frames,
                    reference_frames=reference_frames,
                    max_side=int(runtime.get("max_side", 768)),
                    jpeg_quality=int(runtime.get("jpeg_quality", 85)),
                    detail=str(runtime.get("detail", "low")),
                ),
            }
        ],
        "temperature": float(runtime.get("temperature", 0.0)),
        "max_output_tokens": int(runtime.get("max_output_tokens", 800)),
    }
    endpoint = str(runtime.get("api_base", "https://api.openai.com/v1/responses"))
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    raw_response = _read_http_json(request, int(runtime.get("timeout_seconds", 120)))
    return {
        "provider": "openai",
        "model": body["model"],
        "raw_response": raw_response,
        "response_text": _openai_text_from_response(raw_response),
    }


def _call_gemini(
    *,
    prompt: str,
    prediction_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    api_key = _load_api_key(runtime, default_env="GEMINI_API_KEY")
    model = str(runtime.get("model", "gemini-2.5-pro"))
    body = {
        "contents": [
            {
                "role": "user",
                "parts": _gemini_parts(
                    prompt=prompt,
                    prediction_frames=prediction_frames,
                    reference_frames=reference_frames,
                    max_side=int(runtime.get("max_side", 768)),
                    jpeg_quality=int(runtime.get("jpeg_quality", 85)),
                ),
            }
        ],
        "generationConfig": {
            "temperature": float(runtime.get("temperature", 0.0)),
            "maxOutputTokens": int(runtime.get("max_output_tokens", 800)),
        },
    }
    endpoint = str(
        runtime.get(
            "api_base",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        )
    )
    if "{api_key}" in endpoint:
        endpoint = endpoint.format(api_key=api_key, model=model)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    raw_response = _read_http_json(request, int(runtime.get("timeout_seconds", 120)))
    return {
        "provider": "gemini",
        "model": model,
        "raw_response": raw_response,
        "response_text": _gemini_text_from_response(raw_response),
    }


def _judge_card(sample: BenchmarkSample) -> dict[str, Any]:
    spec = dict(sample.physics_spec or {})
    card = spec.get("judge_card")
    if isinstance(card, dict):
        return card
    group = str(spec.get("group", sample.motion_category or "physics")).strip()
    dimension = str(spec.get("dimension", sample.scene or sample.motion_category or "physics")).strip()
    caption = str(spec.get("reference_caption", sample.prompt_current or sample.prompt_target)).strip()
    label = spec.get("reference_label")
    return build_judge_card(group=group, dimension=dimension, caption=caption, label=label if isinstance(label, str) else None)


def _judge_prompt(sample: BenchmarkSample, card: dict[str, Any]) -> str:
    payload = {
        "sample_id": sample.sample_id,
        "prompt_current": sample.prompt_current,
        "prompt_target": sample.prompt_target,
        "motion_category": sample.motion_category,
        "judge_card": card,
        "response_schema": {
            "scores": {
                "phenomenon": "0..1",
                "causality": "0..1",
                "material": "0..1",
                "temporal": "0..1",
                "prompt": "0..1",
            },
            "confidence": "0..1",
            "failures": ["short tags"],
            "reasoning": "brief rationale",
        },
    }
    return (
        "You are evaluating whether a generated video follows a physical phenomenon rubric.\n"
        "Judge the generated prediction frames against the requested physical behavior, not against visual style alone.\n"
        "Use the reference frames only to understand the intended phenomenon and salient physical pattern.\n"
        "Return strict JSON only, with no markdown and no extra text.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _score_payload(parsed: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    scores = parsed.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    normalized_scores = {
        "phenomenon": _clamp01(scores.get("phenomenon")),
        "causality": _clamp01(scores.get("causality")),
        "material": _clamp01(scores.get("material")),
        "temporal": _clamp01(scores.get("temporal")),
        "prompt": _clamp01(scores.get("prompt")),
    }
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    raw_weights = card.get("weights")
    if isinstance(raw_weights, dict):
        for key in list(weights):
            if key in raw_weights:
                try:
                    weights[key] = float(raw_weights[key])
                except Exception:
                    pass
    weight_sum = sum(max(value, 0.0) for value in weights.values()) or 1.0
    overall = sum(normalized_scores[key] * max(weights[key], 0.0) for key in normalized_scores) / weight_sum
    failures = parsed.get("failures")
    if not isinstance(failures, list):
        failures = []
    return {
        "physics_judge_overall": overall,
        "physics_judge_phenomenon": normalized_scores["phenomenon"],
        "physics_judge_causality": normalized_scores["causality"],
        "physics_judge_material": normalized_scores["material"],
        "physics_judge_temporal": normalized_scores["temporal"],
        "physics_judge_prompt": normalized_scores["prompt"],
        "confidence": _clamp01(parsed.get("confidence")),
        "failures": [str(item).strip() for item in failures if str(item).strip()],
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "weights": weights,
        "scores": normalized_scores,
    }


def evaluate_physics_judge(
    *,
    sample: BenchmarkSample,
    prediction: LoadedMedia,
    reference: LoadedMedia,
    backend: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    cache_root = Path(str(runtime.get("cache_dir", ".cache/worldarena/physics_judge"))).resolve()
    cache_id = _cache_key(sample=sample, prediction=prediction, backend=backend, runtime=runtime)
    cache_path = cache_root / f"{cache_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict):
            cached["cache_hit"] = True
            return cached

    card = _judge_card(sample)
    prompt = _judge_prompt(sample, card)
    prediction_frames = _sample_frames(prediction, int(runtime.get("frame_limit", 6)))
    use_reference = bool(runtime.get("use_reference", True))
    reference_frames = _sample_frames(reference, int(runtime.get("reference_frame_limit", 4))) if use_reference else []
    try:
        if backend == "openai_judge":
            provider_payload = _call_openai(
                prompt=prompt,
                prediction_frames=prediction_frames,
                reference_frames=reference_frames,
                runtime=runtime,
            )
        elif backend == "gemini_judge":
            provider_payload = _call_gemini(
                prompt=prompt,
                prediction_frames=prediction_frames,
                reference_frames=reference_frames,
                runtime=runtime,
            )
        else:
            raise ValueError(f"unsupported physics judge backend: {backend}")
        parsed = _extract_json_object(provider_payload["response_text"])
        scores = _score_payload(parsed, card)
        payload = {
            **scores,
            "success": True,
            "backend": backend,
            "provider": provider_payload["provider"],
            "model": provider_payload["model"],
            "judge_card": card,
            "provider_response_text": provider_payload["response_text"],
            "provider_raw_response": provider_payload["raw_response"],
            "cache_hit": False,
        }
    except (ValueError, KeyError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        payload = {
            "success": False,
            "backend": backend,
            "judge_card": card,
            "error": str(exc),
            "cache_hit": False,
        }
    write_json(cache_path, payload)
    return payload


__all__ = [
    "PHYSICS_JUDGE_METRICS",
    "evaluate_physics_judge",
    "is_physics_judge_metric",
]
