from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INTERMEDIATE_DIR = Path("artifacts/navigation_benchmark/intermediate")
DEFAULT_OUTPUT_DIR = Path("artifacts/navigation_benchmark")
DEFAULT_VIDEO_ROOT = Path("/bench-workspace/yesliu/hjd/data/videos_test_split_4971")
DEFAULT_MIN_COUNT = 200


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scene_phrase(item: dict[str, Any]) -> str:
    environment = str(item.get("environment") or "scene")
    scene = str(item.get("scene") or "space").replace("_", " ")
    if item.get("modality") == "video":
        return f"{environment} {scene} route"
    caption = str(item.get("caption_summary") or "").strip()
    if caption:
        return caption.rstrip(".")
    return f"{environment} {scene}"


def _navigation_scene_type(item: dict[str, Any]) -> str:
    scene = item.get("scene")
    if scene == "passageways":
        return "indoor_corridor"
    if scene in {"public_spaces", "workspaces"}:
        return "indoor_public"
    if scene in {"living_spaces", "dining_spaces"}:
        return "indoor_room"
    if scene in {"city", "suburb"}:
        return "city_walk"
    if scene in {"verdant_landscape", "terrestrial_landscape"}:
        return "outdoor_path"
    return "landscape_route"


def _navigation_template(scene_type: str) -> dict[str, Any]:
    templates = {
        "indoor_corridor": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "push_in", "tilt_up", "push_in", "pan_left"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "forward", "camera_up", "forward", "camera_l"],
            "phases": [
                "enter the visible corridor axis",
                "inspect one side wall or window line",
                "continue deeper without crossing walls",
                "scan the opposite side",
                "advance toward the far landmark",
                "briefly inspect ceiling or upper structure",
                "continue forward and keep the route coherent",
            ],
            "target": "same corridor or passage continues with coherent doors, windows, floor perspective, and architectural landmarks",
        },
        "indoor_public": {
            "tokens": ["push_in", "pan_left", "move_right", "push_in", "pan_right", "move_left", "push_in", "tilt_up"],
            "actions": ["forward", "camera_l", "right", "forward", "camera_r", "left", "forward", "camera_up"],
            "phases": [
                "move along the open aisle or lobby route",
                "look toward shelves, counters, columns, or seating",
                "make a small lateral adjustment around obstacles",
                "continue toward the deepest visible area",
                "scan the other side for landmarks",
                "recentre on the main route",
            ],
            "target": "a larger public interior remains spatially consistent while adjacent aisles, counters, seating, or rooms become visible",
        },
        "indoor_room": {
            "tokens": ["push_in", "pan_left", "orbit_right", "pan_right", "move_left", "push_in", "tilt_down", "pan_left"],
            "actions": ["forward", "camera_l", "forward_right", "camera_r", "left", "forward", "camera_down", "camera_l"],
            "phases": [
                "move gently into the room",
                "look across major furniture and wall landmarks",
                "orbit around the open center without clipping objects",
                "reveal connected room edges or doorway areas",
                "settle on a plausible continuation view",
            ],
            "target": "the same room expands into plausible adjacent corners, doorways, or connected living/dining areas",
        },
        "city_walk": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "move_right", "push_in", "pan_left", "tilt_up"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "right", "forward", "camera_l", "camera_up"],
            "phases": [
                "follow the sidewalk or street perspective",
                "scan storefronts, facades, vehicles, or street furniture",
                "advance along the curb-safe route",
                "look toward the opposite side or next block",
                "continue into the deeper street view",
            ],
            "target": "the city street continues into a coherent block with stable facades, sidewalks, road markings, and urban landmarks",
        },
        "outdoor_path": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "tilt_down", "push_in", "pan_left", "push_in"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "camera_down", "forward", "camera_l", "forward"],
            "phases": [
                "follow the visible trail, bridge, boardwalk, or path",
                "scan vegetation and side landmarks",
                "continue along the route with natural parallax",
                "inspect the ground/path boundary",
                "advance toward the next bend or opening",
            ],
            "target": "the outdoor path continues through coherent vegetation, terrain, railings, sky, and route geometry",
        },
        "landscape_route": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "tilt_down", "push_in", "tilt_up", "pan_left"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "camera_down", "forward", "camera_up", "camera_l"],
            "phases": [
                "move only along a believable visible route",
                "scan terrain and horizon landmarks",
                "keep the ground plane coherent",
                "reveal more of the route ahead without turning the scene into a flat panorama",
            ],
            "target": "a plausible traversable landscape route remains visible with stable terrain and horizon cues",
        },
    }
    return templates.get(scene_type, templates["landscape_route"])


def _navigation_prompt(item: dict[str, Any], template: dict[str, Any]) -> str:
    scene = _scene_phrase(item)
    phases = "; ".join(template["phases"])
    return (
        f"Starting from the provided image of {scene}, generate one continuous first-person exploration video. "
        f"Follow this route intent: {phases}. Preserve the original layout, material style, lighting, and stable landmarks. "
        "Use smooth camera motion, no cuts, no teleportation, and no impossible wall/object clipping."
    )


def _navigation_detection(item: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    scene = _scene_phrase(item)
    return {
        "expected_scene_continuation": template["target"],
        "positive_cues": [
            "visible forward parallax along the walkable floor, path, aisle, sidewalk, or corridor",
            "landmarks from the first frame remain geometrically plausible when revisited or seen from nearby angles",
            "newly revealed regions match the same environment category and visual style",
            "camera follows the specified route intent without teleportation or scene cuts",
        ],
        "failure_modes": [
            "flat zoom with no 3D exploration",
            "camera clips through walls, shelves, furniture, trees, railings, or vehicles",
            "scene identity changes into an unrelated location",
            "major landmarks melt, duplicate, or drift inconsistently",
        ],
        "vlm_checklist": [
            f"Does the video remain a coherent continuation of {scene}?",
            "Is there a clear navigable route rather than only a static zoom?",
            "Are the revealed areas consistent with the starting image style and layout?",
            "Are there cuts, teleportation artifacts, or impossible geometry?",
        ],
        "automated_signals": [
            "DINO/CLIP consistency over sampled frames for landmark and style preservation",
            "optical-flow magnitude to reject static rollouts",
            "VLM route-following score using the positive cues and failure modes above",
        ],
    }


def _transition_type(item: dict[str, Any]) -> str:
    text = " ".join(
        str(item.get(key) or "")
        for key in ["caption_summary", "caption_description", "relative_path", "scene"]
    ).lower()
    if re.search(r"\b(shop|store|cafe|restaurant|diner|gelato|counter|storefront)\b", text):
        return "enter_shop"
    if re.search(r"\b(torii|gate|gateway|archway|arch)\b", text):
        return "pass_gate"
    if re.search(r"\b(stair|stairs|staircase|escalator|tunnel|subway|mine)\b", text):
        return "stairs_or_tunnel"
    if re.search(r"\b(street corner|corner|intersection|alley|side street)\b", text):
        return "street_corner"
    if re.search(r"\b(corridor|hallway|passage|skybridge)\b", text):
        return "corridor_to_room"
    if re.search(r"\b(door|doorway|exit|front door)\b", text):
        return "exit_room" if item.get("environment") == "indoor" else "enter_building"
    if re.search(r"\b(path|trail|bridge|garden|courtyard)\b", text):
        return "path_reveal"
    if item.get("environment") == "outdoor":
        return "street_corner" if item.get("scene") in {"city", "suburb"} else "path_reveal"
    return "corridor_to_room"


def _transition_template(kind: str) -> dict[str, Any]:
    templates = {
        "exit_room": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "tilt_up"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "camera_up"],
            "new_scene": "an exterior porch, front walk, garden, courtyard, or residential street connected to the doorway",
            "boundary": "doorway or exit threshold",
        },
        "enter_shop": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "tilt_down"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "camera_down"],
            "new_scene": "a shop, cafe, diner, or retail interior with counters, shelves, seating, signage, and warm interior detail",
            "boundary": "shopfront or entrance",
        },
        "enter_building": {
            "tokens": ["push_in", "tilt_up", "push_in", "pan_left", "pan_right"],
            "actions": ["forward", "camera_up", "forward", "camera_l", "camera_r"],
            "new_scene": "a lobby, atrium, hall, gallery, or interior landing directly beyond the entrance",
            "boundary": "building entrance or arched doorway",
        },
        "pass_gate": {
            "tokens": ["push_in", "pan_right", "push_in", "pan_left", "tilt_up"],
            "actions": ["forward", "camera_r", "forward", "camera_l", "camera_up"],
            "new_scene": "an inner courtyard, garden, shrine area, patio, or smaller passage revealed after the gate",
            "boundary": "gate, arch, or framed portal",
        },
        "corridor_to_room": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "move_right"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "right"],
            "new_scene": "a connected room, side hall, landing, gallery, office area, or wider interior space",
            "boundary": "corridor opening or room threshold",
        },
        "street_corner": {
            "tokens": ["push_in", "pan_right", "push_in", "pan_left", "tilt_up"],
            "actions": ["forward", "forward_right", "forward", "camera_l", "camera_up"],
            "new_scene": "a cross street, side lane, storefront row, plaza edge, or deeper urban block around the corner",
            "boundary": "street corner, alley mouth, or side-street opening",
        },
        "stairs_or_tunnel": {
            "tokens": ["push_in", "tilt_up", "push_in", "pan_left", "pan_right"],
            "actions": ["forward", "camera_up", "forward", "camera_l", "camera_r"],
            "new_scene": "an upper/lower landing, station concourse, vaulted chamber, platform, or larger connected hall",
            "boundary": "stairs, escalator, tunnel mouth, or long passage end",
        },
        "path_reveal": {
            "tokens": ["push_in", "pan_left", "push_in", "pan_right", "tilt_down"],
            "actions": ["forward", "camera_l", "forward", "camera_r", "camera_down"],
            "new_scene": "a garden clearing, bridge landing, path bend, courtyard, trail continuation, or landscape opening",
            "boundary": "path bend, bridge, garden opening, or natural passage",
        },
    }
    return templates.get(kind, templates["corridor_to_room"])


def _transition_prompt(item: dict[str, Any], kind: str, template: dict[str, Any]) -> str:
    scene = _scene_phrase(item)
    return (
        f"Starting from the provided image of {scene}, generate a continuous first-person scene transition. "
        f"Approach the visible {template['boundary']}, cross it naturally, and reveal {template['new_scene']}. "
        "After the reveal, briefly survey the new area from left to right while preserving spatial continuity with the start image. "
        "Do not jump-cut, do not teleport, and do not replace the scene with an unrelated location."
    )


def _transition_detection(item: dict[str, Any], kind: str, template: dict[str, Any]) -> dict[str, Any]:
    scene = _scene_phrase(item)
    return {
        "transition_type": kind,
        "start_scene": scene,
        "expected_new_scene": template["new_scene"],
        "boundary_to_cross": template["boundary"],
        "positive_cues": [
            f"the camera approaches and crosses the visible {template['boundary']}",
            f"final frames show {template['new_scene']}",
            "the old and new areas are spatially connected rather than separated by a cut",
            "lighting, material style, and perspective evolve plausibly across the boundary",
        ],
        "failure_modes": [
            "the video only pans in place and never reveals a new scene",
            "the boundary remains closed or flat like a painted backdrop",
            "the new scene is unrelated to the entrance/path/corner in the first frame",
            "there is an obvious jump cut, teleport, or sudden style reset",
        ],
        "vlm_checklist": [
            f"Is the initial {template['boundary']} visible or strongly implied?",
            f"Does the generated video cross that {template['boundary']}?",
            f"Do final frames contain {template['new_scene']}?",
            "Is there continuous geometry between start and revealed scene?",
        ],
        "positive_text_queries": [
            template["new_scene"],
            f"camera passes through {template['boundary']} into a new connected scene",
        ],
        "negative_text_queries": [
            "static pan without entering a new place",
            "unrelated scene after a hard cut",
            "closed wall or blocked route",
        ],
        "automated_signals": [
            "VLM yes/no score over sampled middle and final frames using the checklist",
            "CLIP text-image similarity between final frames and positive_text_queries",
            "scene-change evidence: final frames should differ from the first frame while preserving style and continuity",
        ],
    }


def _control_block(prompt: str, tokens: list[str], actions: list[str]) -> dict[str, Any]:
    return {
        "canonical": {
            "duration_policy": "generate at least 60 seconds; distribute ordered steps smoothly over the full rollout",
            "ordered_camera_tokens": tokens,
            "trajectory_granularity": "high-level ordered controls, not per-second timing",
        },
        "yume": {
            "prompt": prompt,
            "control_note": "YUME currently receives image + text; use the prompt as the full navigation contract.",
        },
        "lingbot_world": {
            "camera_path": tokens,
            "pose_synthesis_note": "synthesize poses/intrinsics over the requested output length; do not bind steps to exact seconds",
        },
        "matrix_game": {
            "actions": actions,
            "action_distribution": "spread actions uniformly or by model-native horizon; maintain order but not exact timestamps",
            "mouse_schema": ["pitch_delta", "yaw_delta"],
        },
    }


def _navigation_tasks(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    tasks = []
    for index, item in enumerate(candidates[:count]):
        scene_type = _navigation_scene_type(item)
        template = _navigation_template(scene_type)
        prompt = _navigation_prompt(item, template)
        tasks.append(
            {
                "task_id": f"nav_{index:03d}_{str(item['sheet_id']).lower()}",
                "task_type": "long_navigation",
                "source": _source_block(item),
                "conditioning": _conditioning_block(item),
                "navigation_design": {
                    "scene_type": scene_type,
                    "route_phases": template["phases"],
                    "expected_scene_continuation": template["target"],
                    "prompt": prompt,
                },
                "controls": _control_block(prompt, template["tokens"], template["actions"]),
                "detection": _navigation_detection(item, template),
            }
        )
    return tasks


def _video_navigation_tasks(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    tasks = []
    for index, item in enumerate(candidates[:count]):
        scene_type = _navigation_scene_type(item)
        template = _navigation_template(scene_type)
        prompt = _navigation_prompt(item, template)
        video_prompt = (
            f"{prompt} The source video is a short real navigation/reference clip; use its extracted conditioning frame as "
            "conditioning and continue with the same kind of spatially coherent camera exploration."
        )
        source = _source_block(item)
        source["reference_video_path"] = item["path"]
        source["preview_path"] = item.get("preview_path")
        source["duration_seconds"] = item.get("duration_seconds")
        tasks.append(
            {
                "task_id": f"nav_video_{index:03d}_{str(item['sheet_id']).lower()}",
                "task_type": "long_navigation",
                "source": source,
                "conditioning": _conditioning_block(item),
                "navigation_design": {
                    "scene_type": scene_type,
                    "route_phases": template["phases"],
                    "expected_scene_continuation": template["target"],
                    "prompt": video_prompt,
                    "reference_usage": "source video provides route/style evidence; generation target is still a 60s+ continuation",
                },
                "controls": _control_block(video_prompt, template["tokens"], template["actions"]),
                "detection": {
                    **_navigation_detection(item, template),
                    "reference_video_checks": [
                        "generated early motion should be compatible with the source video camera direction",
                        "generated scene should preserve source video environment type, materials, and route geometry",
                        "long rollout should extend the route rather than simply replay the short source clip",
                    ],
                },
            }
        )
    return tasks


def _transition_tasks(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    tasks = []
    for index, item in enumerate(candidates[:count]):
        kind = _transition_type(item)
        template = _transition_template(kind)
        prompt = _transition_prompt(item, kind, template)
        detection = _transition_detection(item, kind, template)
        tasks.append(
            {
                "task_id": f"scene_transition_{index:03d}_{str(item['sheet_id']).lower()}",
                "task_type": "scene_transition",
                "source": _source_block(item),
                "transition_design": {
                    "transition_type": kind,
                    "start_scene": detection["start_scene"],
                    "boundary_to_cross": detection["boundary_to_cross"],
                    "expected_new_scene": detection["expected_new_scene"],
                    "prompt": prompt,
                },
                "controls": _control_block(prompt, template["tokens"], template["actions"]),
                "detection": detection,
            }
        )
    return tasks


def _memory_tasks(navigation_tasks: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    tokens = ["pan_right"] * 12
    actions = ["camera_r"] * 12
    prompt = (
        "Starting from the provided image, rotate in place clockwise through a full 360 degrees. "
        "Do not translate. Show different directions during the turn, then return to the same starting view "
        "with the same major landmarks, layout, lighting, and object placement."
    )
    tasks = []
    for index, nav_task in enumerate(navigation_tasks[:count]):
        tasks.append(
            {
                "task_id": f"memory_revisit_{index:03d}_{nav_task['task_id'].split('_')[-1]}",
                "task_type": "memory_revisit_360",
                "source": nav_task["source"],
                "conditioning": nav_task.get("conditioning"),
                "memory_design": {
                    "base_navigation_task": nav_task["task_id"],
                    "prompt": prompt,
                    "trajectory_goal": "one full in-place 360-degree rotation with start/end closure",
                    "expected_mid_rollout": "quarter-turn and half-turn views should visibly differ from the start",
                    "expected_final_view": "final frames should match the starting view semantically and geometrically",
                },
                "controls": _control_block(prompt, tokens, actions),
                "detection": {
                    "primary_metric": "memory_revisit_consistency",
                    "positive_cues": [
                        "middle frames show real viewpoint change rather than a static hold",
                        "final frames return to the starting view and preserve major landmarks",
                        "lighting, layout, and object placement remain consistent after the loop",
                    ],
                    "failure_modes": [
                        "static video with no rotation",
                        "rotation never closes back to the initial view",
                        "landmarks disappear, duplicate, or change identity after the turn",
                    ],
                    "automated_signals": [
                        "DINO similarity between first and last frames",
                        "SSIM and MSE between first and last frames",
                        "quarter-turn DINO displacement as a progress gate",
                    ],
                },
            }
        )
    return tasks


def _source_block(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "modality",
        "dataset",
        "path",
        "relative_path",
        "motion_regime",
        "style",
        "environment",
        "scene",
        "width",
        "height",
        "caption_summary",
        "duration_bucket",
    ]
    source = {key: item.get(key) for key in keys if item.get(key) is not None}
    source.setdefault("modality", "image")
    return source


def _conditioning_block(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("modality") == "video":
        return {
            "type": "source_video_frame_image",
            "image_path": item.get("conditioning_image_path"),
            "frame_ratio": item.get("conditioning_frame_ratio"),
            "reference_video_path": item.get("path"),
            "note": "Use the extracted early navigation frame for image-conditioned models; keep the video as route/reference metadata.",
        }
    return {
        "type": "image",
        "image_path": item.get("path"),
    }


def _video_candidates(video_root: Path, output_dir: Path, count: int) -> list[dict[str, Any]]:
    manifest_path = video_root / "_review" / "full_gallery" / "review_manifest.json"
    manifest = _read_json(manifest_path)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in manifest:
        source_rel = str(item.get("source_rel") or "")
        parts = source_rel.split("/")
        if len(parts) < 6 or parts[0] != "static" or parts[1] != "photorealistic":
            continue
        if item.get("prechecked_discard"):
            continue
        duration = float(item.get("duration_seconds") or 0.0)
        if duration < 5.0:
            continue
        environment, scene, duration_bucket = parts[2], parts[3], parts[4]
        if scene not in {
            "passageways",
            "public_spaces",
            "city",
            "suburb",
            "workspaces",
            "living_spaces",
            "verdant_landscape",
            "terrestrial_landscape",
        }:
            continue
        source_path = video_root / source_rel
        preview_path = video_root / "_review" / "full_gallery" / str(item.get("preview_rel") or "")
        if not source_path.exists():
            continue
        candidate = {
            "modality": "video",
            "dataset": "videos_test_split_4971",
            "path": str(source_path),
            "relative_path": source_rel,
            "motion_regime": "static",
            "style": "photorealistic",
            "environment": environment,
            "scene": scene,
            "duration_bucket": duration_bucket,
            "duration_seconds": round(duration, 3),
            "preview_path": str(preview_path) if preview_path.exists() else None,
            "caption_summary": f"{environment} {scene.replace('_', ' ')} navigation video",
        }
        by_key.setdefault((environment, scene), []).append(candidate)

    for bucket in by_key.values():
        bucket.sort(key=lambda row: (-float(row["duration_seconds"]), row["relative_path"]))

    quotas = [
        (("indoor", "passageways"), 42),
        (("outdoor", "city"), 38),
        (("outdoor", "public_spaces"), 30),
        (("outdoor", "suburb"), 24),
        (("indoor", "workspaces"), 22),
        (("indoor", "living_spaces"), 16),
        (("outdoor", "verdant_landscape"), 16),
        (("outdoor", "terrestrial_landscape"), 12),
    ]
    selected: list[dict[str, Any]] = []
    for key, quota in quotas:
        selected.extend(_take_even(by_key.get(key, []), quota))

    if len(selected) < count:
        seen_paths = {item["path"] for item in selected}
        fallback = [
            item
            for bucket in by_key.values()
            for item in bucket
            if item["path"] not in seen_paths
        ]
        fallback.sort(key=lambda row: (-float(row["duration_seconds"]), row["relative_path"]))
        selected.extend(fallback[: count - len(selected)])

    selected = selected[:count]
    first_frame_dir = output_dir / "video_conditioning_frames"
    first_frame_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(selected):
        item["sheet_id"] = f"V{index:03d}"
        conditioning_frame = first_frame_dir / f"{item['sheet_id'].lower()}.jpg"
        _extract_conditioning_frame(Path(item["path"]), conditioning_frame, frame_ratio=0.25)
        item["conditioning_image_path"] = str(conditioning_frame)
        item["conditioning_frame_ratio"] = 0.25
        if conditioning_frame.exists():
            with Image.open(conditioning_frame) as image:
                item["width"], item["height"] = image.size
    return selected


def _take_even(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) <= count:
        return [dict(item) for item in items]
    if count <= 1:
        return [dict(items[0])]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [dict(items[index]) for index in indices]


def _extract_conditioning_frame(video_path: Path, output_path: Path, *, frame_ratio: float) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 1:
            target = int(round(max(0.0, min(float(frame_ratio), 1.0)) * (frame_count - 1)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path, quality=92)


def _font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def _write_contact_sheets(tasks: list[dict[str, Any]], output_dir: Path, prefix: str) -> None:
    for stale in output_dir.glob(f"{prefix}_*.jpg"):
        stale.unlink()
    cols = 6
    sheet_size = 48
    thumb = (220, 150)
    label_h = 48
    font_big = _font("DejaVuSans-Bold.ttf", 18)
    font = _font("DejaVuSans.ttf", 12)
    for start in range(0, len(tasks), sheet_size):
        subset = tasks[start : start + sheet_size]
        rows = math.ceil(len(subset) / cols)
        canvas = Image.new("RGB", (cols * thumb[0], rows * (thumb[1] + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, task in enumerate(subset):
            row, col = divmod(offset, cols)
            x = col * thumb[0]
            y = row * (thumb[1] + label_h)
            image_path = Path(task.get("conditioning", {}).get("image_path") or task["source"]["path"])
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail(thumb, Image.Resampling.LANCZOS)
                background = Image.new("RGB", thumb, (235, 235, 235))
                background.paste(image, ((thumb[0] - image.width) // 2, (thumb[1] - image.height) // 2))
                canvas.paste(background, (x, y))
            except Exception:
                pass
            label = task["task_id"].split("_")[-1].upper()
            draw.rectangle((x, y, x + 88, y + 25), fill=(255, 230, 0))
            draw.text((x + 4, y), label, fill="black", font=font_big)
            scene = str(task["source"].get("scene", "")).replace("_", " ")
            draw.text((x + 4, y + thumb[1] + 2), f"{task['source'].get('dataset', '')} {scene}", fill="black", font=font)
            design = task.get("transition_design") or task.get("navigation_design") or task.get("memory_design") or {}
            summary = str(design.get("transition_type") or design.get("scene_type") or design.get("trajectory_goal") or "")
            draw.text((x + 4, y + thumb[1] + 20), summary[:34], fill="black", font=font)
        end = start + len(subset) - 1
        canvas.save(output_dir / f"{prefix}_{start:03d}_{end:03d}.jpg", quality=92)


def _readme(nav_count: int, image_nav_count: int, video_nav_count: int, transition_count: int, memory_count: int) -> str:
    return f"""# Navigation Benchmark Pack

Generated from stratified visual candidates under:

- `/bench-workspace/yesliu/hjd/data/image_remainder_balanced_4518`
- `/bench-workspace/yesliu/hjd/data/image_test_balanced_5000`

Final task files:

- `navigation_tasks.json`: {nav_count} long navigation cases, combining image and video sources.
- `navigation_image_tasks.json`: {image_nav_count} image-source navigation cases from `image_remainder_balanced_4518`.
- `navigation_video_tasks.json`: {video_nav_count} video-source navigation cases from `videos_test_split_4971`.
- `scene_transition_tasks.json`: {transition_count} new-scene transition cases from both image roots.
- `memory_revisit_tasks.json`: {memory_count} 360-degree revisit cases derived from navigation images.
- `memory_metric_design.md`: start/end revisit metric definition and reporting guidance.

The manifests use high-level ordered controls and prompts, not per-second timing. Each task includes:

- model prompt
- intended route or new-scene target
- canonical camera tokens
- Lingbot and Matrix-Game conversion hints
- detection checklist, positive cues, failure modes, and automated scoring signals
"""


def build_pack(intermediate_dir: Path, output_dir: Path, min_count: int) -> None:
    nav_candidates = _read_json(intermediate_dir / "stratified_nav_candidates.json")
    transition_candidates = _read_json(intermediate_dir / "stratified_transition_candidates.json")
    if len(nav_candidates) < min_count:
        raise ValueError(f"need at least {min_count} navigation candidates, got {len(nav_candidates)}")
    if len(transition_candidates) < min_count:
        raise ValueError(f"need at least {min_count} transition candidates, got {len(transition_candidates)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_navigation_tasks = _navigation_tasks(nav_candidates, min_count)
    video_candidates = _video_candidates(DEFAULT_VIDEO_ROOT, output_dir, min_count)
    video_navigation_tasks = _video_navigation_tasks(video_candidates, min_count)
    navigation_tasks = [*image_navigation_tasks, *video_navigation_tasks]
    transition_tasks = _transition_tasks(transition_candidates, min_count)
    memory_tasks = _memory_tasks(image_navigation_tasks, min_count)

    _write_json(output_dir / "navigation_tasks.json", navigation_tasks)
    _write_json(output_dir / "navigation_image_tasks.json", image_navigation_tasks)
    _write_json(output_dir / "navigation_video_tasks.json", video_navigation_tasks)
    _write_json(output_dir / "scene_transition_tasks.json", transition_tasks)
    _write_json(output_dir / "memory_revisit_tasks.json", memory_tasks)
    (output_dir / "README.md").write_text(
        _readme(
            len(navigation_tasks),
            len(image_navigation_tasks),
            len(video_navigation_tasks),
            len(transition_tasks),
            len(memory_tasks),
        ),
        encoding="utf-8",
    )
    _write_contact_sheets(navigation_tasks, output_dir, "selected_navigation")
    _write_contact_sheets(image_navigation_tasks, output_dir, "selected_navigation_image")
    _write_contact_sheets(video_navigation_tasks, output_dir, "selected_navigation_video")
    _write_contact_sheets(transition_tasks, output_dir, "selected_scene_transition")
    for stale in [output_dir / "selected_navigation_contact_sheet.jpg", output_dir / "selected_scene_transition_contact_sheet.jpg"]:
        if stale.exists():
            stale.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selected navigation benchmark task manifests.")
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    args = parser.parse_args()
    build_pack(args.intermediate_dir, args.output_dir, args.min_count)


if __name__ == "__main__":
    main()
