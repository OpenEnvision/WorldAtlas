from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import requests


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ANNOTATION_FILES = (
    "caption.json",
    "instructions.json",
    "indexes.txt",
    "intrinsics.npy",
    "poses.npy",
    "dyn_masks.npz",
)
INSTRUCTION_LABELS = (
    "Stay",
    "Dolly In",
    "Dolly Out",
    "Pan Left",
    "Pan Right",
    "Truck Left",
    "Truck Right",
    "Tilt Up",
    "Tilt Down",
    "Roll CW",
    "Roll CCW",
    "Pedestal Up",
    "Pedestal Down",
)
INSTRUCTION_LABELS_BY_KEY = {label.lower(): label for label in INSTRUCTION_LABELS}
SCENE_TYPE_DEFAULTS = {
    ("outdoor", "city"): ("Urban", "City"),
    ("outdoor", "public_spaces"): ("Public Space", "Outdoor Public Space"),
    ("outdoor", "suburb"): ("Suburban", "Suburb"),
    ("outdoor", "verdant_landscape"): ("Natural Landscape", "Verdant Landscape"),
    ("outdoor", "terrestrial_landscape"): ("Landscape", "Terrestrial Landscape"),
    ("outdoor", "aquatic_landscape"): ("Aquatic", "Aquatic Landscape"),
    ("indoor", "dining_spaces"): ("Indoor", "Dining Space"),
    ("indoor", "living_spaces"): ("Indoor", "Living Space"),
    ("indoor", "passageways"): ("Indoor", "Passageway"),
    ("indoor", "workspaces"): ("Indoor", "Workspace"),
}
MOTION_CATEGORY_LABELS = {
    "articulated": "Articulated Motion",
    "deformable": "Deformable Motion",
    "fluid": "Fluid Motion",
    "multi_motion": "Multiple Motion Types",
    "rigid": "Rigid Motion",
}
TAG_VALUE_MAP = {
    "bright": "Bright",
    "day": "Daytime",
    "daytime": "Daytime",
    "clear": "Clear",
    "empty": "Empty",
    "none": "Empty",
    "sparse": "Sparse",
    "moderate": "Moderate",
    "dense": "Dense",
    "night": "Night",
    "nighttime": "Night",
    "dusk": "Dusk/Evening",
    "evening": "Dusk/Evening",
    "sunny": "Sunny",
    "cloudy": "Cloudy",
    "rainy": "Rainy",
    "snowy": "Snowy",
    "foggy": "Foggy",
    "stormy": "Stormy",
    "indoor": "Indoor",
    "outdoor": "Outdoor",
    "indoor/unspecified": "Indoor/Unspecified",
    "indoor unspecified": "Indoor/Unspecified",
}
OPENAI_URL = "https://api.openai.com/v1/responses"


@dataclass(slots=True)
class VideoTarget:
    video_path: Path
    annotation_dir: Path
    motion_regime: str
    style: str
    environment: str | None = None
    scene: str | None = None
    duration_bucket: str | None = None
    motion_category: str | None = None


@dataclass(slots=True)
class VideoMetadata:
    frame_count: int
    width: int
    height: int
    fps: float


@dataclass(slots=True)
class SampledFrame:
    frame_index: int
    image_data_url: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing photorealistic WorldArena video annotations with ViPE geometry and GPT-4o text."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/bench-workspace/yesliu/hjd/data/videos"),
        help="Root containing static/ and dynamic/ video trees.",
    )
    parser.add_argument(
        "--vipe-root",
        type=Path,
        default=Path("/bench-workspace/yesliu/hjd/model/WorldEvals/thirdparty/vipe"),
        help="ViPE repository root.",
    )
    parser.add_argument(
        "--openai-key-path",
        type=Path,
        default=Path("/bench-workspace/yesliu/hjd/key"),
        help="Path to a file containing the OpenAI API key.",
    )
    parser.add_argument(
        "--caption-model",
        default="gpt-4o",
        help="OpenAI model used for caption generation.",
    )
    parser.add_argument(
        "--caption-detail",
        choices=("low", "high"),
        default="low",
        help="Image detail level for GPT-4o frame inputs.",
    )
    parser.add_argument(
        "--caption-frame-samples",
        type=int,
        default=6,
        help="Number of evenly spaced frames sent to GPT-4o per video.",
    )
    parser.add_argument(
        "--caption-max-side",
        type=int,
        default=768,
        help="Resize sampled frames so the longest side is at most this many pixels before upload.",
    )
    parser.add_argument(
        "--caption-max-output-tokens",
        type=int,
        default=900,
        help="Max output tokens for each GPT-4o response.",
    )
    parser.add_argument(
        "--caption-timeout-seconds",
        type=int,
        default=180,
        help="HTTP timeout for OpenAI requests.",
    )
    parser.add_argument(
        "--caption-retries",
        type=int,
        default=4,
        help="Number of OpenAI retries per video.",
    )
    parser.add_argument(
        "--regime",
        choices=("all", "static", "dynamic"),
        default="all",
        help="Restrict processing to one motion regime.",
    )
    parser.add_argument(
        "--video",
        action="append",
        type=Path,
        default=[],
        help="Explicit video path(s) to process. If omitted, all missing photorealistic videos are scanned.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process after filtering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite annotations even when the bundle already looks complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matching targets and exit.",
    )
    parser.add_argument(
        "--gpus",
        default="1,3",
        help="Comma-separated GPU ids for parent mode. Child workers should use --gpu instead.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Single GPU id visible to this worker.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=None,
        help="Shard index for child worker mode.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Number of shards for child worker mode.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(".cache/worldarena/vipe_openai_backfill"),
        help="Directory for worker logs.",
    )
    return parser.parse_args(argv)


def configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_gpu_ids(raw_value: str) -> list[str]:
    return [gpu.strip() for gpu in raw_value.split(",") if gpu.strip()]


def is_complete_annotation(annotation_dir: Path) -> bool:
    return annotation_dir.exists() and all((annotation_dir / name).exists() for name in ANNOTATION_FILES)


def should_skip_relative_path(relative_path: Path) -> bool:
    return any(part.startswith(".") or part.startswith("_backup") for part in relative_path.parts)


def title_case_slug(value: str | None) -> str:
    if not value:
        return "Unknown"
    return value.replace("_", " ").replace("-", " ").title()


def build_static_target(video_path: Path, data_root: Path) -> VideoTarget | None:
    root = (data_root / "static").resolve()
    resolved_path = video_path.resolve()
    try:
        relative_path = resolved_path.relative_to(root)
    except ValueError:
        return None
    if should_skip_relative_path(relative_path):
        return None
    parts = relative_path.parts
    if len(parts) < 5:
        return None
    style, environment, scene, duration_bucket, _ = parts[:5]
    if style != "photorealistic":
        return None
    annotation_dir = resolved_path.parent / "annotations" / f"{resolved_path.stem}_SpatialVID_ann"
    return VideoTarget(
        video_path=resolved_path,
        annotation_dir=annotation_dir,
        motion_regime="static",
        style=style,
        environment=environment,
        scene=scene,
        duration_bucket=duration_bucket,
    )


def build_dynamic_target(video_path: Path, data_root: Path) -> VideoTarget | None:
    root = (data_root / "dynamic").resolve()
    resolved_path = video_path.resolve()
    try:
        relative_path = resolved_path.relative_to(root)
    except ValueError:
        return None
    if should_skip_relative_path(relative_path):
        return None
    parts = relative_path.parts
    if len(parts) < 4 or parts[1] != "videos":
        return None
    style, _, motion_category, _ = parts[:4]
    if style != "photorealistic":
        return None
    annotation_dir = root / "annotations" / motion_category / f"{resolved_path.stem}_ann"
    return VideoTarget(
        video_path=resolved_path,
        annotation_dir=annotation_dir,
        motion_regime="dynamic",
        style=style,
        motion_category=motion_category,
    )


def build_target(video_path: Path, data_root: Path) -> VideoTarget | None:
    return build_static_target(video_path, data_root) or build_dynamic_target(video_path, data_root)


def iter_video_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def discover_targets(args: argparse.Namespace) -> list[VideoTarget]:
    data_root = args.data_root
    candidates: list[VideoTarget] = []
    if args.video:
        for video_path in sorted(set(args.video)):
            target = build_target(video_path, data_root)
            if target is None:
                logging.warning("Skipping unsupported path: %s", video_path)
                continue
            candidates.append(target)
    else:
        if args.regime in {"all", "static"}:
            for video_path in iter_video_paths(data_root / "static"):
                target = build_static_target(video_path, data_root)
                if target is not None:
                    candidates.append(target)
        if args.regime in {"all", "dynamic"}:
            for video_path in iter_video_paths(data_root / "dynamic"):
                target = build_dynamic_target(video_path, data_root)
                if target is not None:
                    candidates.append(target)

    filtered: list[VideoTarget] = []
    for target in candidates:
        if not args.force and is_complete_annotation(target.annotation_dir):
            continue
        filtered.append(target)
        if args.limit is not None and len(filtered) >= args.limit:
            break
    return filtered


def read_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"video frame count is invalid: {video_path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"video resolution is invalid: {video_path}")
    return VideoMetadata(frame_count=frame_count, width=width, height=height, fps=fps)


def sample_frame_indices(frame_count: int, num_samples: int) -> list[int]:
    num = max(1, min(num_samples, frame_count))
    if num == 1:
        return [0]
    indices = np.linspace(0, frame_count - 1, num=num, dtype=np.int32)
    return sorted({int(index) for index in indices.tolist()})


def resize_frame(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame_bgr
    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def encode_frame_to_data_url(frame_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode sampled frame as JPEG")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def sample_video_frames(video_path: Path, metadata: VideoMetadata, num_samples: int, max_side: int) -> list[SampledFrame]:
    frame_indices = sample_frame_indices(metadata.frame_count, num_samples)
    capture = cv2.VideoCapture(str(video_path))
    sampled_frames: list[SampledFrame] = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video for frame sampling: {video_path}")
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to sample frame {frame_index} from {video_path}")
            resized = resize_frame(frame, max_side=max_side)
            sampled_frames.append(
                SampledFrame(
                    frame_index=frame_index,
                    image_data_url=encode_frame_to_data_url(resized),
                )
            )
    finally:
        capture.release()
    return sampled_frames


def relative_pose_summary(poses: np.ndarray) -> dict[str, Any]:
    if len(poses) < 2:
        return {
            "translation_xyz": [0.0, 0.0, 0.0],
            "rotation_xyz_deg": [0.0, 0.0, 0.0],
            "path_length": 0.0,
        }
    start_pose = poses[0].astype(np.float64)
    end_pose = poses[-1].astype(np.float64)
    delta = np.linalg.inv(start_pose) @ end_pose
    rotation = delta[:3, :3]
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if singular:
        rot_x = math.atan2(-rotation[1, 2], rotation[1, 1])
        rot_y = math.atan2(-rotation[2, 0], sy)
        rot_z = 0.0
    else:
        rot_x = math.atan2(rotation[2, 1], rotation[2, 2])
        rot_y = math.atan2(-rotation[2, 0], sy)
        rot_z = math.atan2(rotation[1, 0], rotation[0, 0])
    path_length = float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()) if len(poses) > 1 else 0.0
    return {
        "translation_xyz": [round(float(value), 4) for value in delta[:3, 3]],
        "rotation_xyz_deg": [round(math.degrees(value), 3) for value in (rot_x, rot_y, rot_z)],
        "path_length": round(path_length, 4),
    }


def dominant_instruction_labels_from_pose(poses: np.ndarray) -> list[str]:
    summary = relative_pose_summary(poses)
    translation = np.asarray(summary["translation_xyz"], dtype=np.float32)
    rotation = np.asarray(summary["rotation_xyz_deg"], dtype=np.float32)
    if float(np.linalg.norm(translation)) < 0.04 and float(np.max(np.abs(rotation))) < 2.0:
        return ["Stay"]

    labels: list[str] = []
    translation_axis = int(np.argmax(np.abs(translation)))
    if float(np.max(np.abs(translation))) >= 0.08:
        if translation_axis == 2:
            labels.append("Dolly Out" if translation[2] > 0 else "Dolly In")
        elif translation_axis == 0:
            labels.append("Truck Left" if translation[0] > 0 else "Truck Right")
        else:
            labels.append("Pedestal Down" if translation[1] > 0 else "Pedestal Up")

    if float(np.max(np.abs(rotation))) >= 3.0 and len(labels) < 2:
        rotation_axis = int(np.argmax(np.abs(rotation)))
        if rotation_axis == 1:
            labels.append("Pan Right" if rotation[1] > 0 else "Pan Left")
        elif rotation_axis == 0:
            labels.append("Tilt Down" if rotation[0] > 0 else "Tilt Up")
        else:
            labels.append("Roll CW" if rotation[2] > 0 else "Roll CCW")

    return labels or ["Stay"]


def scene_type_defaults(target: VideoTarget) -> tuple[str, str]:
    if target.motion_regime == "dynamic":
        return ("Dynamic", MOTION_CATEGORY_LABELS.get(target.motion_category or "", title_case_slug(target.motion_category)))
    key = (target.environment or "", target.scene or "")
    if key in SCENE_TYPE_DEFAULTS:
        return SCENE_TYPE_DEFAULTS[key]
    if target.environment == "indoor":
        return ("Indoor", title_case_slug(target.scene))
    if target.environment == "outdoor":
        return ("Outdoor", title_case_slug(target.scene))
    return ("Unknown", title_case_slug(target.scene))


def default_category_tags(target: VideoTarget) -> dict[str, Any]:
    primary_scene, secondary_scene = scene_type_defaults(target)
    weather = "Indoor/Unspecified" if target.environment == "indoor" else "Unknown"
    return {
        "sceneType": {
            "first": primary_scene,
            "second": secondary_scene,
        },
        "brightness": "Unknown",
        "timeOfDay": "Unknown",
        "weather": weather,
        "crowdDensity": "Unknown",
    }


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def normalize_tag_value(value: Any, *, fallback: str, slug_to_title: bool = False) -> str:
    text = clean_text(value)
    if not text:
        return fallback
    normalized = TAG_VALUE_MAP.get(text.lower())
    if normalized:
        return normalized
    if slug_to_title:
        return title_case_slug(text)
    return text


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened: list[str] = []
        for nested in value:
            flattened.extend(flatten_strings(nested))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for nested in value.values():
            flattened.extend(flatten_strings(nested))
        return flattened
    return []


def normalize_instruction_labels(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        for raw_piece in str(value).replace("|", ",").split(","):
            piece = " ".join(raw_piece.strip().split())
            if not piece:
                continue
            key = piece.lower()
            if key in INSTRUCTION_LABELS_BY_KEY:
                label = INSTRUCTION_LABELS_BY_KEY[key]
                if label not in normalized:
                    normalized.append(label)
    return normalized


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response is not a JSON object")
    return payload


def extract_output_text(response_payload: dict[str, Any]) -> str:
    text_chunks: list[str] = []
    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                text_chunks.append(content["text"])
    output_text = "\n".join(text_chunks).strip()
    if not output_text:
        raise ValueError("response did not contain assistant output_text")
    return output_text


def build_openai_prompt(target: VideoTarget, metadata: VideoMetadata, pose_summary: dict[str, Any]) -> str:
    context_lines = [
        "You are writing benchmark annotations for one single video clip.",
        "The sampled images that follow are ordered from earliest frame to latest frame.",
        "Use the visual evidence from the images as the main source of truth.",
        "Use the dataset metadata and ViPE motion summary only as supporting context.",
        "Return valid JSON only. Do not wrap the JSON in markdown.",
        "",
        "Required JSON shape:",
        json.dumps(
            {
                "caption": {
                    "SceneSummary": "string",
                    "SceneDescription": "string",
                    "CameraMotion": "string",
                    "ShotImmersion": "string",
                    "CategoryTags": {
                        "sceneType": {
                            "first": "string",
                            "second": "string",
                        },
                        "brightness": "string",
                        "timeOfDay": "string",
                        "weather": "string",
                        "crowdDensity": "string",
                    },
                    "MotionTrends": ["string", "string", "string"],
                },
                "instructions": {
                    f"0->{metadata.frame_count - 1}": ["Dolly In"]
                },
            },
            ensure_ascii=True,
        ),
        "",
        "Rules:",
        "- SceneSummary: exactly one sentence grounded in what is visible.",
        "- SceneDescription: 1-3 sentences about the visible setting, salient subjects, and motion.",
        "- CameraMotion: one sentence describing the camera movement over the clip.",
        "- ShotImmersion: one sentence about how the shot feels to a viewer.",
        "- MotionTrends: 2-4 short strings.",
        "- instructions: use camera-motion labels only, not object-motion labels.",
        f"- The only allowed instruction labels are: {', '.join(INSTRUCTION_LABELS)}.",
        f"- The instructions segment key must be exactly '0->{metadata.frame_count - 1}'.",
        "- Do not mention uncertainty, frame sampling, or metadata in the output.",
        "",
        f"Known dataset regime: {target.motion_regime}",
        f"Known style bucket: {target.style}",
    ]
    if target.motion_regime == "static":
        context_lines.extend(
            [
                f"Known environment bucket: {target.environment}",
                f"Known scene bucket: {target.scene}",
                f"Known duration bucket: {target.duration_bucket}",
            ]
        )
    else:
        context_lines.append(f"Known motion category bucket: {target.motion_category}")
    context_lines.extend(
        [
            f"Video resolution: {metadata.width}x{metadata.height}",
            f"Video fps: {metadata.fps:.3f}",
            f"Video frame count: {metadata.frame_count}",
            f"ViPE translation summary (relative first->last): {pose_summary['translation_xyz']}",
            f"ViPE rotation summary in degrees (relative first->last, xyz): {pose_summary['rotation_xyz_deg']}",
            f"ViPE camera path length: {pose_summary['path_length']}",
        ]
    )
    return "\n".join(context_lines)


def call_openai_annotation(
    *,
    target: VideoTarget,
    metadata: VideoMetadata,
    sampled_frames: list[SampledFrame],
    pose_summary: dict[str, Any],
    api_key: str,
    session: requests.Session,
    model: str,
    detail: str,
    max_output_tokens: int,
    timeout_seconds: int,
    retries: int,
) -> dict[str, Any]:
    prompt = build_openai_prompt(target, metadata, pose_summary)
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for position, sampled_frame in enumerate(sampled_frames, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Chronological sample {position} of {len(sampled_frames)}. Original video frame index: {sampled_frame.frame_index}.",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": sampled_frame.image_data_url,
                "detail": detail,
            }
        )

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
        "temperature": 0.2,
        "store": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            request_id = response.headers.get("x-request-id")
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                body = response.text[:1000]
                logging.warning(
                    "OpenAI transient error on %s (attempt %d/%d, status=%s, request_id=%s): %s",
                    target.video_path.name,
                    attempt,
                    retries,
                    response.status_code,
                    request_id,
                    body,
                )
                if attempt == retries:
                    response.raise_for_status()
                time.sleep(min(60.0, 4.0 * attempt))
                continue
            response.raise_for_status()
            response_payload = response.json()
            if response_payload.get("status") != "completed":
                raise RuntimeError(
                    f"OpenAI response did not complete for {target.video_path.name}: {response_payload.get('status')}"
                )
            output_text = extract_output_text(response_payload)
            return extract_json_object(output_text)
        except requests.RequestException as exc:
            logging.warning(
                "OpenAI request failure on %s (attempt %d/%d): %s",
                target.video_path.name,
                attempt,
                retries,
                exc,
            )
            if attempt == retries:
                raise
            time.sleep(min(60.0, 4.0 * attempt))
        except Exception as exc:
            logging.warning(
                "Failed to parse OpenAI JSON for %s (attempt %d/%d, request_id=%s): %s",
                target.video_path.name,
                attempt,
                retries,
                request_id if "request_id" in locals() else None,
                exc,
            )
            if attempt == retries:
                raise
            time.sleep(min(60.0, 4.0 * attempt))
    raise RuntimeError(f"unreachable retry exhaustion for {target.video_path}")


def normalize_caption_payload(raw_payload: dict[str, Any], target: VideoTarget) -> dict[str, Any]:
    caption = raw_payload.get("caption", raw_payload)
    if not isinstance(caption, dict):
        raise ValueError("caption payload is not an object")

    scene_summary = clean_text(caption.get("SceneSummary"))
    scene_description = clean_text(caption.get("SceneDescription"))
    camera_motion = clean_text(caption.get("CameraMotion"))
    shot_immersion = clean_text(caption.get("ShotImmersion"))
    if not all((scene_summary, scene_description, camera_motion, shot_immersion)):
        raise ValueError("caption payload is missing required text fields")

    default_tags = default_category_tags(target)
    category_tags = caption.get("CategoryTags")
    if not isinstance(category_tags, dict):
        category_tags = {}
    scene_type = category_tags.get("sceneType")
    if not isinstance(scene_type, dict):
        scene_type = {}
    normalized_category_tags = {
        "sceneType": {
            "first": normalize_tag_value(
                scene_type.get("first"),
                fallback=default_tags["sceneType"]["first"],
                slug_to_title=True,
            ),
            "second": normalize_tag_value(
                scene_type.get("second"),
                fallback=default_tags["sceneType"]["second"],
                slug_to_title=True,
            ),
        },
        "brightness": normalize_tag_value(category_tags.get("brightness"), fallback=default_tags["brightness"]),
        "timeOfDay": normalize_tag_value(category_tags.get("timeOfDay"), fallback=default_tags["timeOfDay"]),
        "weather": normalize_tag_value(category_tags.get("weather"), fallback=default_tags["weather"]),
        "crowdDensity": normalize_tag_value(category_tags.get("crowdDensity"), fallback=default_tags["crowdDensity"]),
    }

    motion_trends = [clean_text(item) for item in flatten_strings(caption.get("MotionTrends"))]
    motion_trends = [item for item in motion_trends if item]
    if not motion_trends:
        motion_trends = [camera_motion]
    if target.motion_regime == "dynamic":
        motion_label = MOTION_CATEGORY_LABELS.get(target.motion_category or "", title_case_slug(target.motion_category))
        if motion_label not in motion_trends:
            motion_trends.append(motion_label)
    motion_trends = motion_trends[:4]

    return {
        "SceneSummary": scene_summary,
        "SceneDescription": scene_description,
        "CameraMotion": camera_motion,
        "ShotImmersion": shot_immersion,
        "CategoryTags": normalized_category_tags,
        "MotionTrends": motion_trends,
    }


def normalize_instructions_payload(
    raw_payload: dict[str, Any],
    metadata: VideoMetadata,
    poses: np.ndarray,
) -> dict[str, list[str]]:
    segment_key = f"0->{metadata.frame_count - 1}"
    raw_instructions = raw_payload.get("instructions", raw_payload.get("Instructions"))
    values: list[str] = []
    if isinstance(raw_instructions, dict):
        for segment_value in raw_instructions.values():
            values.extend(flatten_strings(segment_value))
            break
    elif raw_instructions is not None:
        values.extend(flatten_strings(raw_instructions))
    labels = normalize_instruction_labels(values)
    if not labels:
        labels = dominant_instruction_labels_from_pose(poses)
    return {segment_key: labels[:2]}


def write_indexes(index_path: Path, frame_indices: np.ndarray) -> None:
    lines = [f"# total {len(frame_indices)} indexes"]
    lines.extend(f"{item_index} {int(frame_index)}" for item_index, frame_index in enumerate(frame_indices.tolist()))
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_zero_dyn_masks(mask_path: Path, metadata: VideoMetadata) -> None:
    payload: dict[str, Any] = {"shape": np.asarray([metadata.height, metadata.width], dtype=np.int32)}
    zero_data = np.zeros((0,), dtype=np.uint8)
    zero_indices = np.zeros((0,), dtype=np.int32)
    zero_indptr = np.zeros((metadata.height + 1,), dtype=np.int32)
    for frame_index in range(metadata.frame_count):
        payload[f"f_{frame_index}_data"] = zero_data
        payload[f"f_{frame_index}_indices"] = zero_indices
        payload[f"f_{frame_index}_indptr"] = zero_indptr
    np.savez_compressed(mask_path, **payload)


def prepare_annotation_dir(annotation_dir: Path) -> Path:
    temp_dir = annotation_dir.parent / f".{annotation_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def finalize_annotation_dir(temp_dir: Path, annotation_dir: Path) -> None:
    annotation_dir.parent.mkdir(parents=True, exist_ok=True)
    if annotation_dir.exists():
        annotation_dir.mkdir(parents=True, exist_ok=True)
        for path in temp_dir.iterdir():
            os.replace(path, annotation_dir / path.name)
        temp_dir.rmdir()
    else:
        os.replace(temp_dir, annotation_dir)


def initialize_vipe_pipeline(vipe_root: Path):
    if str(vipe_root) not in sys.path:
        sys.path.insert(0, str(vipe_root))

    import hydra
    from vipe import get_config_path, make_pipeline

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose(
            "default",
            overrides=[
                "pipeline=default",
                "pipeline.init.instance=null",
                "pipeline.slam.keyframe_depth=null",
                "pipeline.post.depth_align_model=null",
                "pipeline.output.path=/tmp/worldarena_vipe_backfill",
                "pipeline.output.save_artifacts=false",
                "pipeline.output.save_viz=false",
                "pipeline.output.save_slam_map=false",
            ],
        )

    pipeline = make_pipeline(args.pipeline)
    pipeline.return_output_streams = True
    return pipeline


def run_vipe_geometry(vipe_root: Path, pipeline: Any, video_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if str(vipe_root) not in sys.path:
        sys.path.insert(0, str(vipe_root))

    import torch
    from vipe.streams.base import FrameAttribute
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    stream = RawMp4Stream(video_path)
    output = pipeline.run(stream)
    if not output.output_streams:
        raise RuntimeError(f"ViPE returned no output stream for {video_path}")
    output_stream = output.output_streams[0]
    pose_values = output_stream.get_stream_attribute(FrameAttribute.POSE)
    intrinsics_values = output_stream.get_stream_attribute(FrameAttribute.INTRINSICS)

    pose_by_index = {
        frame_index: pose.matrix().cpu().numpy().astype(np.float32)
        for frame_index, pose in enumerate(pose_values)
        if pose is not None
    }
    intrinsics_by_index = {
        frame_index: intrinsics.cpu().numpy().astype(np.float32)[:4]
        for frame_index, intrinsics in enumerate(intrinsics_values)
        if intrinsics is not None
    }
    common_indices = sorted(set(pose_by_index) & set(intrinsics_by_index))
    if not common_indices:
        raise RuntimeError(f"ViPE produced no aligned pose/intrinsics frames for {video_path}")

    poses = np.stack([pose_by_index[index] for index in common_indices], axis=0).astype(np.float32)
    intrinsics = np.stack([intrinsics_by_index[index] for index in common_indices], axis=0).astype(np.float32)
    indices = np.asarray(common_indices, dtype=np.int32)
    torch.cuda.empty_cache()
    return poses, intrinsics, indices


def process_target(
    *,
    target: VideoTarget,
    api_key: str,
    session: requests.Session,
    pipeline: Any,
    args: argparse.Namespace,
) -> None:
    logging.info("Processing %s", target.video_path)
    metadata = read_video_metadata(target.video_path)
    poses, intrinsics, frame_indices = run_vipe_geometry(args.vipe_root, pipeline, target.video_path)
    pose_summary = relative_pose_summary(poses)
    sampled_frames = sample_video_frames(
        target.video_path,
        metadata,
        num_samples=args.caption_frame_samples,
        max_side=args.caption_max_side,
    )
    raw_payload = call_openai_annotation(
        target=target,
        metadata=metadata,
        sampled_frames=sampled_frames,
        pose_summary=pose_summary,
        api_key=api_key,
        session=session,
        model=args.caption_model,
        detail=args.caption_detail,
        max_output_tokens=args.caption_max_output_tokens,
        timeout_seconds=args.caption_timeout_seconds,
        retries=args.caption_retries,
    )
    caption_payload = normalize_caption_payload(raw_payload, target)
    instructions_payload = normalize_instructions_payload(raw_payload, metadata, poses)

    temp_dir = prepare_annotation_dir(target.annotation_dir)
    try:
        (temp_dir / "caption.json").write_text(
            json.dumps(caption_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temp_dir / "instructions.json").write_text(
            json.dumps(instructions_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_indexes(temp_dir / "indexes.txt", frame_indices)
        np.save(temp_dir / "poses.npy", poses.astype(np.float32))
        np.save(temp_dir / "intrinsics.npy", intrinsics.astype(np.float32))
        write_zero_dyn_masks(temp_dir / "dyn_masks.npz", metadata)
        finalize_annotation_dir(temp_dir, target.annotation_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def read_api_key(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"OpenAI key file is empty: {path}")
    return value


def run_worker(args: argparse.Namespace) -> int:
    worker_label = f"worker_{args.worker_index}" if args.worker_index is not None else "worker_single"
    gpu_label = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    log_path = args.log_dir / f"{worker_label}_gpu_{gpu_label}.log"
    configure_logging(log_path)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    targets = discover_targets(args)
    if args.worker_count:
        shard_index = args.worker_index or 0
        targets = [target for index, target in enumerate(targets) if index % args.worker_count == shard_index]

    logging.info(
        "Starting worker with %d target(s); gpu=%s; regime=%s; force=%s",
        len(targets),
        gpu_label,
        args.regime,
        args.force,
    )
    if args.dry_run:
        for target in targets:
            logging.info("Dry run target: %s -> %s", target.video_path, target.annotation_dir)
        return 0
    if not targets:
        logging.info("No targets to process.")
        return 0

    api_key = read_api_key(args.openai_key_path)
    pipeline = initialize_vipe_pipeline(args.vipe_root)
    session = requests.Session()

    succeeded = 0
    failed = 0
    started_at = time.time()
    for item_index, target in enumerate(targets, start=1):
        logging.info("[%d/%d] %s", item_index, len(targets), target.video_path.name)
        try:
            process_target(
                target=target,
                api_key=api_key,
                session=session,
                pipeline=pipeline,
                args=args,
            )
            succeeded += 1
            logging.info("Completed %s", target.video_path)
        except Exception as exc:
            failed += 1
            logging.exception("Failed to process %s: %s", target.video_path, exc)
    elapsed = time.time() - started_at
    logging.info(
        "Worker finished: succeeded=%d failed=%d elapsed_seconds=%.2f",
        succeeded,
        failed,
        elapsed,
    )
    return 1 if failed else 0


def forward_args_to_child(args: argparse.Namespace, worker_index: int, worker_count: int, gpu: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-root",
        str(args.data_root),
        "--vipe-root",
        str(args.vipe_root),
        "--openai-key-path",
        str(args.openai_key_path),
        "--caption-model",
        args.caption_model,
        "--caption-detail",
        args.caption_detail,
        "--caption-frame-samples",
        str(args.caption_frame_samples),
        "--caption-max-side",
        str(args.caption_max_side),
        "--caption-max-output-tokens",
        str(args.caption_max_output_tokens),
        "--caption-timeout-seconds",
        str(args.caption_timeout_seconds),
        "--caption-retries",
        str(args.caption_retries),
        "--regime",
        args.regime,
        "--log-dir",
        str(args.log_dir),
        "--worker-index",
        str(worker_index),
        "--worker-count",
        str(worker_count),
        "--gpu",
        str(gpu),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    for video_path in args.video:
        command.extend(["--video", str(video_path)])
    return command


def run_parent(args: argparse.Namespace) -> int:
    configure_logging()
    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    if len(gpu_ids) == 1:
        args.gpu = gpu_ids[0]
        return run_worker(args)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for worker_index, gpu in enumerate(gpu_ids):
        command = forward_args_to_child(args, worker_index, len(gpu_ids), gpu)
        worker_log = args.log_dir / f"parent_worker_{worker_index}_gpu_{gpu}.stdout.log"
        logging.info("Launching worker %d on GPU %s", worker_index, gpu)
        handle = worker_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
        processes.append((str(worker_log), process))

    exit_code = 0
    for log_path, process in processes:
        status = process.wait()
        if status != 0:
            exit_code = status
            logging.error("Worker failed: log=%s exit_code=%d", log_path, status)
        else:
            logging.info("Worker finished successfully: log=%s", log_path)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    child_mode = args.worker_count is not None or args.worker_index is not None or args.gpu is not None
    if child_mode:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
