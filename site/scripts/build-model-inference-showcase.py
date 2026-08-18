#!/usr/bin/env python3
"""Build the WorldArena model-inference promotional showcase.

The builder deliberately separates selection from rendering:

1. ``select`` merges canonical per-sample benchmark reports, balances model and
   scene coverage, and writes review contact sheets.
2. ``render`` turns the reviewed selection into a cinematic 1080p showcase.

Only Python's standard library, Pillow, and ffmpeg are required.  All choices
are deterministic so a refreshed prediction tree can be reviewed and rendered
with the same commands.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_ROOT = REPO_ROOT / "artifacts" / "predictions"
FINAL_RESULTS_MANIFEST = REPO_ROOT / "artifacts" / "reports" / "FINAL_RESULTS_MANIFEST.json"
RECONSTRUCTION_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "benchmark"
    / "reports"
    / "reconstruction_consistency_wbench_image_static_20260718"
)
DEFAULT_BUILD_DIR = REPO_ROOT / "artifacts" / "model_inference_showcase"
DEFAULT_OUTPUT = REPO_ROOT / "site" / "assets" / "model_inference_showcase.mp4"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_MONO_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")


STATIC_WEIGHTS = {
    "image_quality": 0.30,
    "perceptual_quality": 0.24,
    "hpsv3_norm": 0.20,
    "sharpness_retention": 0.12,
    "segment_continuity": 0.08,
    "temporal_flickering": 0.06,
}

DYNAMIC_WEIGHTS = {
    "motion_accuracy": 0.34,
    "motion_smoothness": 0.26,
    "temporal_flickering": 0.24,
    "motion_magnitude": 0.16,
}

RECONSTRUCTION_WEIGHTS = {
    "geometric_consistency": 0.58,
    "photometric_consistency": 0.42,
}


MODEL_LABELS = {
    "ac3d-image-dynamic": "AC3D",
    "ac3d-image-static": "AC3D",
    "alayaworld-production": "AlayaWorld",
    "astra_image_static": "Astra",
    "cameractrl_image_dynamic": "CameraCtrl",
    "cameractrl_image_static": "CameraCtrl",
    "cogvideo-x1.5-5b-i2v-image-dynamic": "CogVideoX 1.5",
    "cosmos3-nano-image-dynamic": "Cosmos-Nano",
    "cosmos3-nano-image-static": "Cosmos-Nano",
    "cosmos_predict25_2b_image_static": "Cosmos-Predict2.5",
    "dreamx_world_image_static": "DreamX",
    "flashworld-image-static": "FlashWorld",
    "gen3c-image-static": "GEN3C",
    "hunyuan-gamecraft-image-static": "Hunyuan GameCraft",
    "hunyuan-gamecraft-static-dynamic": "Hunyuan GameCraft",
    "hunyuan-video-i2v-image-static-dynamic": "HunyuanVideo I2V",
    "hunyuanworld-voyager-image-static-129f": "HunyuanWorld Voyager",
    "hy_worldplay_ar_distilled_image_test_balanced_5000_5s": "HY-WorldPlay Distilled",
    "hy_worldplay_ar_image_test_balanced_5000_5s": "HY-WorldPlay AR",
    "hy_worldplay_ar_rl_image_test_balanced_5000_5s": "HY-WorldPlay RL",
    "infinite_world_image_static_5s": "InfiniteWorld",
    "lingbot_world_image_dynamic": "LingBot-World",
    "lingbot_world_image_static": "LingBot-World",
    "lingbot_world_official_8gpu_image_static": "LingBot-World Official",
    "longcat-video-image-dynamic-8gpu-20260612_133600": "LongCat-Video",
    "longcat-video-image-static-8gpu-20260612_133950": "LongCat-Video",
    "ltx23-image-static-8gpu-full_20260612_182624": "LTX-Video 2.3",
    "ltx23-image-static-8gpu-full_official_default_20260613_031436": "LTX-Video 2.3",
    "luciddreamer-image-static-8gpu-20260613_151844": "LucidDreamer",
    "lyra-2-image-static-openenvision": "Lyra 2",
    "matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2": "Matrix-Game 1",
    "matrix-game-3-image-dynamic-5s": "Matrix-Game 3",
    "matrix_game2_5s_image_dynamic": "Matrix-Game 2",
    "matrix_game2_5s_image_static": "Matrix-Game 2",
    "matrix_game3_5s_image_static_dynamic": "Matrix-Game 3.5",
    "motionctrl_image_static_official_default_tmp2": "MotionCtrl",
    "sana-wm-image-dynamic": "Sana-WM",
    "sana-wm-image-static": "Sana-WM",
    "stable-virtual-camera-image-dynamic-5s": "Stable Virtual Camera",
    "stable_virtual_camera_image_static_5s": "Stable Virtual Camera",
    "stable_virtual_camera_image_static_worldscore_v2": "Stable Virtual Camera V2",
    "wan22-image-static-8gpu-20260612_160808": "Wan 2.2",
    "wan22-ti2v-5b-image-static-plan-missing": "Wan TI2V-5B",
    "warp-as-history-image-static-8gpu-20260612_152000": "Warp-as-History",
    "wonderjourney-image-static-8gpu-singleload-wj_full_20260614_043849": "WonderJourney",
    "wonderjourney-image-static-wj_full-viewable": "WonderJourney",
    "wonderworld-image-static-8gpu-singleload-20260614_8gpu_image_static_v3": "WonderWorld",
    "wonderworld-image-static-v3-viewable": "WonderWorld",
    "worldfm-image-static-121f-8gpu-20260702_060713": "WorldFM",
    "yume-sample5b-image-static-dynamic": "YUME",
    "yume_sample5b_image_static_dynamic_tmp": "YUME",
}


EXTRA_RECONSTRUCTION_MODELS = (
    "cosmos_predict25_2b_image_static",
    "gen3c-image-static",
    "hunyuan-gamecraft-image-static",
    "hunyuan-video-i2v-image-static-dynamic",
    "hunyuanworld-voyager-image-static-129f",
    "infinite_world_image_static_5s",
    "lingbot_world_official_8gpu_image_static",
    "ltx23-image-static-8gpu-full_20260612_182624",
    "wan22-image-static-8gpu-20260612_160808",
    "wan22-ti2v-5b-image-static-plan-missing",
    "wonderjourney-image-static-wj_full-viewable",
    "yume-sample5b-image-static-dynamic",
)


# Automated metrics are the first pass, not the final editorial decision.
# These entries were rejected from the 2026-08-08 contact-sheet review for
# blank/near-blank frames, severe softness, weak composition, or duplication.
# Keeping the decisions here makes a refreshed render auditable.
MANUAL_REJECTIONS = {
    ("cameractrl_image_dynamic", "image_dynamic_photorealistic_articulated_0093"),
    ("cameractrl_image_dynamic", "image_dynamic_stylized_deformable_0137"),
    ("sana-wm-image-dynamic", "image_dynamic_photorealistic_deformable_0121"),
    ("cosmos_predict25_2b_image_static", "image_static_photorealistic_living_spaces_0059"),
    ("hunyuan-gamecraft-image-static", "image_static_photorealistic_public_spaces_0060"),
    ("hunyuan-gamecraft-static-dynamic", "image_static_photorealistic_public_spaces_0060"),
    ("hunyuan-gamecraft-static-dynamic", "image_static_photorealistic_passageways_0166"),
    ("ltx23-image-static-8gpu-full_20260612_182624", "image_static_photorealistic_public_spaces_0163"),
    ("motionctrl_image_static_official_default_tmp2", "image_static_stylized_living_spaces_0135"),
    ("motionctrl_image_static_official_default_tmp2", "image_static_photorealistic_suburb_0067"),
    ("wonderjourney-image-static-wj_full-viewable", "image_static_photorealistic_living_spaces_0015"),
    ("wonderworld-image-static-v3-viewable", "image_static_photorealistic_aquatic_landscape_0025"),
    ("wonderworld-image-static-v3-viewable", "image_static_photorealistic_living_spaces_0016"),
    ("worldfm-image-static-121f-8gpu-20260702_060713", "image_static_photorealistic_living_spaces_0015"),
    ("worldfm-image-static-121f-8gpu-20260702_060713", "image_static_photorealistic_passageways_0068"),
}


@dataclasses.dataclass
class Clip:
    model: str
    model_label: str
    suite: str
    sample_id: str
    path: str
    score: float
    metrics: dict[str, float]
    style: str = ""
    environment: str = ""
    scene: str = ""
    motion_category: str = ""
    selection_source: str = "canonical"
    review_status: str = "selected"

    def category(self) -> str:
        if self.motion_category:
            return self.motion_category.replace("_", " ")
        if self.scene:
            return self.scene.replace("_", " ")
        return self.suite.replace("_", " ")

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run(command: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ffmpeg_path(explicit: str | None = None) -> str:
    candidates = [
        Path(explicit) if explicit else None,
        REPO_ROOT / ".tools" / "bin" / "ffmpeg",
        Path(shutil.which("ffmpeg")) if shutil.which("ffmpeg") else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    raise FileNotFoundError("ffmpeg was not found; pass --ffmpeg explicitly")


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_prediction_path(recorded: str, model: str, sample_id: str) -> Path:
    path = Path(recorded)
    marker = "/artifacts/predictions/"
    if marker in recorded:
        # Reports were produced from both /mnt/cpfs and /mnt/cpfsB mounts.  Do
        # not stat hundreds of thousands of remote files while merging scores;
        # rebasing is deterministic and the much smaller final selection is
        # validated later by balanced_pick/contact-sheet extraction.
        return PREDICTIONS_ROOT / recorded.split(marker, 1)[1]
    if recorded:
        return path
    return PREDICTIONS_ROOT / model / f"{sample_id}.mp4"


def metric_value(payload: dict[str, Any]) -> float | None:
    value = payload.get("normalized")
    if value is None:
        value = payload.get("raw")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def choose_metric_report(
    reports: list[dict[str, Any]], model: str, suite: str, metric: str
) -> dict[str, Any] | None:
    options = [
        report
        for report in reports
        if report["model_name"] == model
        and report["suite"] == suite
        and metric in report["metrics"]
    ]
    if not options:
        return None
    return max(
        options,
        key=lambda report: (
            report.get("rows", 0),
            -report.get("row_errors", 0),
            report.get("generated_at", ""),
        ),
    )


def load_canonical_pool(
    reports: list[dict[str, Any]], suite: str, weights: dict[str, float]
) -> dict[str, list[Clip]]:
    models = sorted({r["model_name"] for r in reports if r["suite"] == suite})
    result: dict[str, list[Clip]] = {}
    for model in models:
        merged: dict[str, dict[str, Any]] = {}
        for metric in weights:
            report = choose_metric_report(reports, model, suite, metric)
            if not report:
                continue
            score_path = REPO_ROOT / report["report_dir"] / "per_sample_scores.jsonl"
            if not score_path.exists():
                continue
            for row in load_jsonl(score_path):
                sample_id = row["sample_id"]
                entry = merged.setdefault(
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "prediction_path": row.get("prediction_path", ""),
                        "style": row.get("style") or "",
                        "environment": row.get("environment") or "",
                        "scene": row.get("scene") or "",
                        "motion_category": row.get("motion_category") or "",
                        "metrics": {},
                    },
                )
                payload = row.get("metrics", {}).get(metric, {})
                value = metric_value(payload)
                if value is not None:
                    entry["metrics"][metric] = value
                if row.get("prediction_path"):
                    entry["prediction_path"] = row["prediction_path"]

        clips: list[Clip] = []
        for row in merged.values():
            present_weight = sum(weights[m] for m in row["metrics"] if m in weights)
            if not present_weight:
                continue
            score = sum(weights[m] * v for m, v in row["metrics"].items()) / present_weight
            resolved = resolve_prediction_path(row["prediction_path"], model, row["sample_id"])
            clips.append(
                Clip(
                    model=model,
                    model_label=MODEL_LABELS.get(model, model.replace("_", " ")),
                    suite=suite,
                    sample_id=row["sample_id"],
                    path=str(resolved),
                    score=round(score, 6),
                    metrics={k: round(v, 6) for k, v in row["metrics"].items()},
                    style=row["style"],
                    environment=row["environment"],
                    scene=row["scene"],
                    motion_category=row["motion_category"],
                    selection_source="canonical_metrics",
                )
            )
        result[model] = sorted(clips, key=lambda clip: clip.score, reverse=True)
    return result


def load_reconstruction_pool() -> dict[str, list[Clip]]:
    result: dict[str, list[Clip]] = {}
    for model in EXTRA_RECONSTRUCTION_MODELS:
        score_path = RECONSTRUCTION_ROOT / model / "per_sample_scores.jsonl"
        if not score_path.exists():
            continue
        clips: list[Clip] = []
        for row in load_jsonl(score_path):
            metrics: dict[str, float] = {}
            for metric in RECONSTRUCTION_WEIGHTS:
                value = metric_value(row.get("metrics", {}).get(metric, {}))
                if value is not None:
                    metrics[metric] = value
            present_weight = sum(RECONSTRUCTION_WEIGHTS[m] for m in metrics)
            if not present_weight:
                continue
            score = sum(RECONSTRUCTION_WEIGHTS[m] * v for m, v in metrics.items()) / present_weight
            resolved = resolve_prediction_path(row.get("prediction_path", ""), model, row["sample_id"])
            clips.append(
                Clip(
                    model=model,
                    model_label=MODEL_LABELS.get(model, model.replace("_", " ")),
                    suite=row.get("suite", "image_static"),
                    sample_id=row["sample_id"],
                    path=str(resolved),
                    score=round(score, 6),
                    metrics={k: round(v, 6) for k, v in metrics.items()},
                    style=row.get("style") or "",
                    environment=row.get("environment") or "",
                    scene=row.get("scene") or "",
                    motion_category=row.get("motion_category") or "",
                    selection_source="reconstruction_metrics",
                )
            )
        result[model] = sorted(clips, key=lambda clip: clip.score, reverse=True)
    return result


def clip_bucket(clip: Clip) -> str:
    if clip.suite == "image_dynamic":
        return f"{clip.style or 'unknown'}:{clip.motion_category or 'dynamic'}"
    if clip.style == "stylized":
        return "stylized"
    return f"photo:{clip.environment or 'unknown'}:{clip.scene or 'scene'}"


def balanced_pick(pool: list[Clip], count: int) -> list[Clip]:
    """Pick high-scoring clips without taking the same visual class twice."""
    if count <= 0:
        return []
    chosen: list[Clip] = []
    used_buckets: set[str] = set()
    used_samples: set[str] = set()

    # Pass one values broad visual coverage over tiny score differences.
    for clip in pool:
        bucket = clip_bucket(clip)
        if bucket in used_buckets or clip.sample_id in used_samples:
            continue
        if (clip.model, clip.sample_id) in MANUAL_REJECTIONS:
            continue
        if not Path(clip.path).exists():
            continue
        chosen.append(clip)
        used_buckets.add(bucket)
        used_samples.add(clip.sample_id)
        if len(chosen) == count:
            return chosen

    # Pass two fills sparse or partial runs from their strongest remaining clips.
    for clip in pool:
        if (
            clip.sample_id in used_samples
            or (clip.model, clip.sample_id) in MANUAL_REJECTIONS
            or not Path(clip.path).exists()
        ):
            continue
        chosen.append(clip)
        used_samples.add(clip.sample_id)
        if len(chosen) == count:
            break
    return chosen


def load_alayaworld_clips(count: int = 4) -> list[Clip]:
    model = "alayaworld-production"
    paths = sorted((PREDICTIONS_ROOT / model).glob("*.mp4"))
    if not paths:
        return []
    # The production smoke contains eight articulated examples.  Even spacing
    # prevents the default filesystem order from dominating the selection.
    indices = [round(i * (len(paths) - 1) / max(1, count - 1)) for i in range(count)]
    clips: list[Clip] = []
    for rank, index in enumerate(dict.fromkeys(indices)):
        path = paths[index]
        clips.append(
            Clip(
                model=model,
                model_label=MODEL_LABELS[model],
                suite="image_dynamic",
                sample_id=path.stem,
                path=str(path),
                score=round(0.5 - rank * 0.001, 6),
                metrics={},
                style="photorealistic",
                motion_category="articulated",
                selection_source="production_smoke",
            )
        )
    return clips


def comparison_groups(
    static_pool: dict[str, list[Clip]], group_count: int = 4, models_per_group: int = 4
) -> list[dict[str, Any]]:
    by_sample: dict[str, list[Clip]] = defaultdict(list)
    for clips in static_pool.values():
        for clip in clips[:400]:
            by_sample[clip.sample_id].append(clip)

    candidates: list[tuple[float, str, list[Clip]]] = []
    for sample_id, clips in by_sample.items():
        best_by_label: dict[str, Clip] = {}
        for clip in clips:
            old = best_by_label.get(clip.model_label)
            if old is None or clip.score > old.score:
                best_by_label[clip.model_label] = clip
        ranked = sorted(best_by_label.values(), key=lambda clip: clip.score, reverse=True)
        if len(ranked) < models_per_group:
            continue
        selected = ranked[:models_per_group]
        mean_score = sum(clip.score for clip in selected) / models_per_group
        candidates.append((mean_score, sample_id, selected))

    candidates.sort(reverse=True)
    groups: list[dict[str, Any]] = []
    used_categories: set[str] = set()
    used_models: defaultdict[str, int] = defaultdict(int)
    for mean_score, sample_id, clips in candidates:
        category = clips[0].category()
        if category in used_categories:
            continue
        # Rotate over strong models instead of showing the same top four in
        # every comparison panel.
        alternates = sorted(
            by_sample[sample_id],
            key=lambda clip: (used_models[clip.model_label], -clip.score),
        )
        selected: list[Clip] = []
        labels: set[str] = set()
        for clip in alternates:
            if clip.model_label in labels:
                continue
            selected.append(clip)
            labels.add(clip.model_label)
            if len(selected) == models_per_group:
                break
        if len(selected) < models_per_group:
            continue
        groups.append(
            {
                "sample_id": sample_id,
                "category": category,
                "mean_score": round(mean_score, 6),
                "clips": [clip.to_json() for clip in selected],
            }
        )
        used_categories.add(category)
        for clip in selected:
            used_models[clip.model_label] += 1
        if len(groups) == group_count:
            break
    return groups


def save_selection(
    build_dir: Path,
    selected: list[Clip],
    comparisons: list[dict[str, Any]],
    pool_stats: dict[str, Any],
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "selection_policy": {
            "static_weights": STATIC_WEIGHTS,
            "dynamic_weights": DYNAMIC_WEIGHTS,
            "reconstruction_weights": RECONSTRUCTION_WEIGHTS,
            "per_canonical_static_model": 2,
            "per_canonical_dynamic_model": 2,
            "per_extra_reconstruction_model": 2,
            "manual_review_field": "review_status",
        },
        "pool_stats": pool_stats,
        "selected_clip_count": len(selected),
        "selected_model_count": len({clip.model_label for clip in selected}),
        "selected": [clip.to_json() for clip in selected],
        "comparison_groups": comparisons,
    }
    destination = build_dir / "selection.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {destination} ({payload['selected_clip_count']} clips, "
        f"{payload['selected_model_count']} display models)"
    )
    return destination


def extract_thumbnail(ffmpeg: str, clip: Clip, destination: Path) -> tuple[Path, str | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "1.2",
        "-i",
        clip.path,
        "-frames:v",
        "1",
        "-vf",
        "scale=360:202:force_original_aspect_ratio=increase,crop=360:202",
        "-q:v",
        "3",
        str(destination),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode or not destination.exists():
        # Some short or unusual files seek more reliably after opening.
        command[5:7] = []
        command[command.index("-i") + 2 : command.index("-frames:v")] = ["-ss", "0.5"]
        completed = subprocess.run(command, text=True, capture_output=True)
    error = completed.stderr.strip() if completed.returncode or not destination.exists() else None
    return destination, error


def make_contact_sheets(
    ffmpeg: str, build_dir: Path, clips: list[Clip], workers: int = 16
) -> list[Path]:
    thumb_dir = build_dir / "review_thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (clip, thumb_dir / f"{index:03d}.jpg")
        for index, clip in enumerate(clips, start=1)
    ]
    errors: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(extract_thumbnail, ffmpeg, clip, path): index
            for index, (clip, path) in enumerate(jobs, start=1)
        }
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            _, error = future.result()
            if error:
                errors[index] = error

    font_title = ImageFont.truetype(str(FONT_BOLD), 18)
    font_meta = ImageFont.truetype(str(FONT_REGULAR), 13)
    font_index = ImageFont.truetype(str(FONT_MONO_BOLD), 17)
    tile_w, image_h, label_h = 360, 202, 78
    tile_h = image_h + label_h
    columns, rows = 4, 5
    per_page = columns * rows
    sheets: list[Path] = []

    for page_start in range(0, len(clips), per_page):
        page_clips = clips[page_start : page_start + per_page]
        canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), "#080b12")
        draw = ImageDraw.Draw(canvas)
        for offset, clip in enumerate(page_clips):
            index = page_start + offset + 1
            x = (offset % columns) * tile_w
            y = (offset // columns) * tile_h
            thumb_path = thumb_dir / f"{index:03d}.jpg"
            if thumb_path.exists():
                with Image.open(thumb_path) as thumb:
                    canvas.paste(thumb.convert("RGB"), (x, y))
            else:
                draw.rectangle((x, y, x + tile_w, y + image_h), fill="#251018")
                draw.text((x + 12, y + 84), "THUMBNAIL ERROR", font=font_title, fill="#ff7285")
            draw.rectangle((x, y + image_h, x + tile_w, y + tile_h), fill="#111722")
            draw.rectangle((x, y + image_h, x + 5, y + tile_h), fill="#4ee6d4")
            draw.text((x + 12, y + image_h + 8), f"{index:03d}", font=font_index, fill="#ffbd45")
            draw.text((x + 58, y + image_h + 7), clip.model_label[:28], font=font_title, fill="white")
            meta = f"{clip.suite.replace('_', ' ')}  |  {clip.category()}"
            draw.text((x + 12, y + image_h + 35), meta[:49], font=font_meta, fill="#aeb9cc")
            detail = f"score {clip.score:.3f}  |  {clip.style or clip.selection_source}"
            draw.text((x + 12, y + image_h + 55), detail[:52], font=font_meta, fill="#74839a")
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#283244", width=1)
        page_number = page_start // per_page + 1
        destination = build_dir / f"contact_sheet_{page_number:02d}.jpg"
        canvas.save(destination, quality=92, subsampling=0)
        sheets.append(destination)
        print(f"Wrote {destination}")

    if errors:
        error_path = build_dir / "thumbnail_errors.json"
        error_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
        print(f"Thumbnail failures: {len(errors)} (see {error_path})", file=sys.stderr)
    return sheets


HERO_STATIC_PRIORITY = (
    ("ac3d-image-static", "image_static_photorealistic_aquatic_landscape_0079"),
    ("cosmos3-nano-image-static", "image_static_photorealistic_verdant_landscape_0161"),
    ("gen3c-image-static", "image_static_photorealistic_aquatic_landscape_0156"),
    (
        "hy_worldplay_ar_distilled_image_test_balanced_5000_5s",
        "image_static_photorealistic_verdant_landscape_0066",
    ),
    ("hunyuanworld-voyager-image-static-129f", "image_static_photorealistic_city_0008"),
    ("lingbot_world_image_static", "image_static_stylized_city_0159"),
    ("longcat-video-image-static-8gpu-20260612_133950", "image_static_photorealistic_aquatic_landscape_0107"),
    ("luciddreamer-image-static-8gpu-20260613_151844", "image_static_stylized_living_spaces_0122"),
    (
        "matrix-game-1-image-static-worldarena-prompt-min5-lowmem-rmsnorm-v2",
        "image_static_photorealistic_city_0141",
    ),
    ("matrix_game2_5s_image_static", "image_static_photorealistic_terrestrial_landscape_0015"),
    ("sana-wm-image-static", "image_static_photorealistic_city_0156"),
    ("wonderjourney-image-static-wj_full-viewable", "image_static_photorealistic_aquatic_landscape_0131"),
)

HERO_DYNAMIC_PRIORITY = (
    ("ac3d-image-dynamic", "image_dynamic_photorealistic_fluid_0092"),
    ("alayaworld-production", "image_dynamic_photorealistic_articulated_0006"),
    ("cogvideo-x1.5-5b-i2v-image-dynamic", "image_dynamic_photorealistic_articulated_0111"),
    ("cosmos3-nano-image-dynamic", "image_dynamic_stylized_deformable_0167"),
    ("lingbot_world_image_dynamic", "image_dynamic_stylized_fluid_0010"),
    ("longcat-video-image-dynamic-8gpu-20260612_133600", "image_dynamic_stylized_multi_motion_0041"),
    ("matrix_game2_5s_image_dynamic", "image_dynamic_photorealistic_articulated_0034"),
    ("matrix-game-3-image-dynamic-5s", "image_dynamic_photorealistic_articulated_0088"),
    ("sana-wm-image-dynamic", "image_dynamic_photorealistic_rigid_0063"),
    ("stable-virtual-camera-image-dynamic-5s", "image_dynamic_photorealistic_fluid_0112"),
)


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def fit_font(
    draw: ImageDraw.ImageDraw, text: str, path: Path, maximum: int, width: int
) -> ImageFont.FreeTypeFont:
    for size in range(maximum, 15, -2):
        candidate = font(path, size)
        box = draw.textbbox((0, 0), text, font=candidate)
        if box[2] - box[0] <= width:
            return candidate
    return font(path, 16)


def shadow_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int] | str,
    *,
    anchor: str | None = None,
    shadow: int = 3,
) -> None:
    x, y = position
    draw.text((x + shadow, y + shadow), text, font=text_font, fill=(0, 0, 0, 190), anchor=anchor)
    draw.text((x, y), text, font=text_font, fill=fill, anchor=anchor)


def add_corner_branding(draw: ImageDraw.ImageDraw) -> None:
    small = font(FONT_MONO_BOLD, 16)
    draw.rounded_rectangle((1485, 42, 1866, 84), radius=5, fill=(5, 10, 19, 165), outline=(72, 224, 210, 110), width=1)
    draw.ellipse((1501, 54, 1517, 70), outline=rgba("#4EE6D4"), width=2)
    draw.line((1509, 50, 1509, 74), fill=rgba("#FFC857"), width=2)
    draw.text((1528, 54), "WORLD ARENA // MODEL INFERENCE", font=small, fill=rgba("#DDE8F5"))


def save_hero_overlay(clip: Clip, destination: Path) -> None:
    image = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    add_corner_branding(draw)
    draw.rounded_rectangle((54, 832, 778, 1017), radius=10, fill=(5, 9, 17, 178), outline=(94, 116, 143, 90), width=1)
    draw.rectangle((54, 832, 63, 1017), fill=rgba("#4EE6D4"))
    draw.rectangle((54, 832, 286, 838), fill=rgba("#FFC857"))
    draw.text((86, 858), "MODEL INFERENCE", font=font(FONT_MONO_BOLD, 16), fill=rgba("#FFC857"))
    title_font = fit_font(draw, clip.model_label, FONT_BOLD, 44, 640)
    shadow_text(draw, (84, 892), clip.model_label, title_font, rgba("#FFFFFF"), shadow=2)
    metadata = f"{clip.suite.replace('_', ' ').upper()}  //  {clip.category().upper()}"
    draw.text((86, 955), metadata, font=font(FONT_REGULAR, 19), fill=rgba("#AFC2D8"))
    draw.line((54, 1038, 1866, 1038), fill=(255, 255, 255, 35), width=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def save_wall_overlays(
    clips: list[Clip],
    destination_labels: Path,
    destination_title: Path,
    *,
    columns: int,
    rows: int,
    title: str,
    subtitle: str,
    eyebrow: str,
    statistic: str,
    show_labels: bool,
    darken: int = 0,
) -> None:
    labels = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(labels)
    if darken:
        label_draw.rectangle((0, 0, 1920, 1080), fill=(2, 6, 12, darken))
    if show_labels:
        tile_w, tile_h = 1920 // columns, 1080 // rows
        label_font = font(FONT_BOLD, 14 if columns >= 5 else 16)
        meta_font = font(FONT_MONO_BOLD, 10 if columns >= 5 else 11)
        for index, clip in enumerate(clips):
            col, row = index % columns, index // columns
            x, y = col * tile_w, row * tile_h
            bar_h = 45 if rows <= 3 else 38
            label_draw.rectangle((x, y + tile_h - bar_h, x + tile_w, y + tile_h), fill=(4, 8, 15, 158))
            accent = rgba("#4EE6D4") if index % 2 == 0 else rgba("#FFC857")
            label_draw.rectangle((x, y + tile_h - bar_h, x + 5, y + tile_h), fill=accent)
            display = clip.model_label
            if len(display) > 25:
                display = display[:23] + "…"
            label_draw.text((x + 14, y + tile_h - bar_h + 6), display, font=label_font, fill=rgba("#FFFFFF"))
            label_draw.text(
                (x + 14, y + tile_h - 17),
                clip.category().upper()[:31],
                font=meta_font,
                fill=rgba("#9EB0C6"),
            )
    add_corner_branding(label_draw)

    title_layer = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(title_layer)
    card = (330, 365, 1590, 725)
    draw.rounded_rectangle(card, radius=18, fill=(4, 8, 16, 205), outline=(78, 230, 212, 105), width=2)
    draw.rectangle((330, 365, 610, 373), fill=rgba("#FFC857"))
    draw.rectangle((610, 365, 1590, 373), fill=rgba("#4EE6D4"))
    draw.text((960, 412), eyebrow, font=font(FONT_MONO_BOLD, 18), fill=rgba("#FFC857"), anchor="ma")
    title_font = fit_font(draw, title, FONT_BOLD, 68, 1120)
    shadow_text(draw, (960, 488), title, title_font, rgba("#FFFFFF"), anchor="mm", shadow=3)
    draw.text((960, 575), subtitle, font=font(FONT_REGULAR, 25), fill=rgba("#C9D7E7"), anchor="mm")
    draw.line((545, 632, 1375, 632), fill=(255, 255, 255, 50), width=1)
    draw.text((960, 667), statistic, font=font(FONT_MONO_BOLD, 17), fill=rgba("#4EE6D4"), anchor="mm")
    destination_labels.parent.mkdir(parents=True, exist_ok=True)
    labels.save(destination_labels)
    title_layer.save(destination_title)


def save_comparison_overlay(group: dict[str, Any], destination: Path) -> None:
    image = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1920, 104), fill=(4, 8, 15, 215))
    draw.rectangle((0, 101, 1920, 106), fill=rgba("#4EE6D4"))
    draw.text((72, 29), "ONE PROMPT. FOUR MODELS.", font=font(FONT_BOLD, 34), fill=rgba("#FFFFFF"))
    draw.text(
        (1848, 39),
        group["category"].upper(),
        font=font(FONT_MONO_BOLD, 17),
        fill=rgba("#FFC857"),
        anchor="ra",
    )
    for index, payload in enumerate(group["clips"]):
        col, row = index % 2, index // 2
        x, y = col * 960, row * 540
        # Top-row labels sit below the global title bar; bottom-row labels use
        # the same local baseline for a stable comparison grid.
        label_y = y + 467
        draw.rounded_rectangle((x + 26, label_y, x + 410, label_y + 53), radius=6, fill=(4, 8, 15, 185))
        draw.rectangle((x + 26, label_y, x + 33, label_y + 53), fill=rgba("#FFC857" if index % 2 else "#4EE6D4"))
        draw.text((x + 48, label_y + 13), payload["model_label"][:30], font=font(FONT_BOLD, 20), fill=rgba("#FFFFFF"))
    draw.line((960, 106, 960, 1080), fill=(255, 255, 255, 85), width=2)
    draw.line((0, 540, 1920, 540), fill=(255, 255, 255, 85), width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def video_encode_args(threads: int = 8) -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.2",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-threads",
        str(threads),
        "-movflags",
        "+faststart",
    ]


def normalized_filter(index: int, width: int, height: int, duration: float, output: str) -> str:
    return (
        f"[{index}:v]fps=24,"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,"
        "eq=contrast=1.025:saturation=1.055:brightness=-0.004,"
        f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[{output}]"
    )


def render_hero_clip(
    ffmpeg: str,
    clip: Clip,
    destination: Path,
    overlay_path: Path,
    duration: float,
    force: bool,
) -> Path:
    if destination.exists() and not force:
        return destination
    save_hero_overlay(clip, overlay_path)
    filters = [normalized_filter(0, 1920, 1080, duration, "base")]
    filters.append("[1:v]format=rgba[ui]")
    filters.append("[base][ui]overlay=0:0:shortest=1,format=yuv420p[out]")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stream_loop",
        "-1",
        "-i",
        clip.path,
        "-loop",
        "1",
        "-framerate",
        "24",
        "-i",
        str(overlay_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-t",
        f"{duration:.3f}",
        "-r",
        "24",
        "-an",
        *video_encode_args(6),
        str(destination),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(command, quiet=True)
    return destination


def stack_layout(columns: int, rows: int, count: int, tile_w: int, tile_h: int) -> str:
    positions = []
    for index in range(count):
        col, row = index % columns, index // columns
        positions.append(f"{col * tile_w}_{row * tile_h}")
    return "|".join(positions)


def padded(clips: list[Clip], count: int) -> list[Clip]:
    if not clips:
        raise ValueError("cannot pad an empty clip list")
    return [clips[index % len(clips)] for index in range(count)]


def render_wall(
    ffmpeg: str,
    clips: list[Clip],
    destination: Path,
    overlay_dir: Path,
    *,
    columns: int,
    rows: int,
    duration: float,
    title: str,
    subtitle: str,
    eyebrow: str,
    statistic: str,
    title_hold: float,
    show_labels: bool = True,
    darken: int = 0,
    fade_in: float = 0.18,
    fade_out: float = 0.18,
    force: bool = False,
) -> Path:
    if destination.exists() and not force:
        return destination
    count = columns * rows
    clips = padded(clips, count)
    labels_path = overlay_dir / f"{destination.stem}_labels.png"
    title_path = overlay_dir / f"{destination.stem}_title.png"
    save_wall_overlays(
        clips,
        labels_path,
        title_path,
        columns=columns,
        rows=rows,
        title=title,
        subtitle=subtitle,
        eyebrow=eyebrow,
        statistic=statistic,
        show_labels=show_labels,
        darken=darken,
    )
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for clip in clips:
        command.extend(["-stream_loop", "-1", "-i", clip.path])
    for overlay_path in (labels_path, title_path):
        command.extend(["-loop", "1", "-framerate", "24", "-i", str(overlay_path)])

    tile_w, tile_h = 1920 // columns, 1080 // rows
    filters = [
        normalized_filter(index, tile_w, tile_h, duration, f"v{index}")
        for index in range(count)
    ]
    inputs = "".join(f"[v{index}]" for index in range(count))
    filters.append(
        f"{inputs}xstack=inputs={count}:layout={stack_layout(columns, rows, count, tile_w, tile_h)}:fill=0x050912[wall]"
    )
    filters.append(f"[{count}:v]format=rgba[labels]")
    filters.append("[wall][labels]overlay=0:0:shortest=1[labeled]")
    title_fade_out_start = max(0.1, title_hold - 0.45)
    filters.append(
        f"[{count + 1}:v]format=rgba,fade=t=in:st=0.12:d=0.42:alpha=1,"
        f"fade=t=out:st={title_fade_out_start:.3f}:d=0.45:alpha=1[title]"
    )
    filters.append(
        f"[labeled][title]overlay=0:0:enable='between(t,0,{title_hold:.3f})'[composite]"
    )
    tail_start = max(0.0, duration - fade_out)
    filters.append(
        f"[composite]format=yuv420p,fade=t=in:st=0:d={fade_in:.3f},"
        f"fade=t=out:st={tail_start:.3f}:d={fade_out:.3f}[out]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            f"{duration:.3f}",
            "-r",
            "24",
            "-an",
            *video_encode_args(16),
            str(destination),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(command, quiet=True)
    return destination


def render_comparison_group(
    ffmpeg: str,
    group: dict[str, Any],
    destination: Path,
    overlay_path: Path,
    duration: float,
    force: bool,
) -> Path:
    if destination.exists() and not force:
        return destination
    clips = [Clip(**payload) for payload in group["clips"]]
    save_comparison_overlay(group, overlay_path)
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for clip in clips:
        command.extend(["-stream_loop", "-1", "-i", clip.path])
    command.extend(["-loop", "1", "-framerate", "24", "-i", str(overlay_path)])
    filters = [normalized_filter(index, 960, 540, duration, f"v{index}") for index in range(4)]
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540:fill=0x050912[grid]")
    filters.append("[4:v]format=rgba[ui]")
    filters.append("[grid][ui]overlay=0:0:shortest=1,format=yuv420p[out]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            f"{duration:.3f}",
            "-r",
            "24",
            "-an",
            *video_encode_args(12),
            str(destination),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(command, quiet=True)
    return destination


def concat_videos(ffmpeg: str, videos: list[Path], destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        return destination
    list_path = destination.with_suffix(".concat.txt")
    list_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in videos), encoding="utf-8"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        quiet=True,
    )
    return destination


def render_soundtrack(ffmpeg: str, duration: float, destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        return destination
    fade_out = max(0.0, duration - 3.0)
    filters = (
        "[0:a]tremolo=f=2:d=0.92,volume=0.070,lowpass=f=175[bass];"
        "[1:a]tremolo=f=0.25:d=0.72,volume=0.020,aecho=0.8:0.72:420|840:0.16|0.09[mid];"
        "[2:a]tremolo=f=0.125:d=0.58,volume=0.0075,highpass=f=160[air];"
        "[3:a]highpass=f=70,lowpass=f=1800,volume=0.0045[noise];"
        f"[bass][mid][air][noise]amix=inputs=4:duration=longest:normalize=0,"
        "volume=14.0,"
        f"afade=t=in:st=0:d=2.0,afade=t=out:st={fade_out:.3f}:d=3.0,"
        "alimiter=limit=0.90[out]"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=48:sample_rate=48000:duration={duration:.3f}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=96:sample_rate=48000:duration={duration:.3f}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=192:sample_rate=48000:duration={duration:.3f}",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=color=pink:sample_rate=48000:duration={duration:.3f}",
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(destination),
        ],
        quiet=True,
    )
    return destination


def diverse_order(clips: list[Clip]) -> list[Clip]:
    groups: dict[str, list[Clip]] = defaultdict(list)
    for clip in sorted(clips, key=lambda item: item.score, reverse=True):
        groups[clip.model_label].append(clip)
    ordered: list[Clip] = []
    while groups:
        for label in list(sorted(groups)):
            ordered.append(groups[label].pop(0))
            if not groups[label]:
                del groups[label]
    return ordered


def priority_pick(clips: list[Clip], priorities: Iterable[tuple[str, str]], count: int) -> list[Clip]:
    lookup = {(clip.model, clip.sample_id): clip for clip in clips}
    chosen = [lookup[key] for key in priorities if key in lookup]
    used = {clip.path for clip in chosen}
    for clip in diverse_order(clips):
        if clip.path not in used:
            chosen.append(clip)
            used.add(clip.path)
        if len(chosen) >= count:
            break
    return chosen[:count]


def split_chunks(items: list[Clip], size: int, count: int) -> list[list[Clip]]:
    output: list[list[Clip]] = []
    cursor = 0
    for _ in range(count):
        output.append(padded(items[cursor : cursor + size] or items, size))
        cursor += size
    return output


def render_hero_sequence(
    ffmpeg: str,
    clips: list[Clip],
    destination: Path,
    work_dir: Path,
    duration_per_clip: float,
    workers: int,
    force: bool,
) -> Path:
    clip_dir = work_dir / f"{destination.stem}_clips"
    overlay_dir = work_dir / "overlays"
    outputs = [clip_dir / f"{index:02d}_{clip.model_label.lower().replace(' ', '_')}.mp4" for index, clip in enumerate(clips)]
    print(f"Rendering {destination.stem}: {len(clips)} full-screen hero cuts", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        futures = [
            executor.submit(
                render_hero_clip,
                ffmpeg,
                clip,
                output,
                overlay_dir / f"{destination.stem}_{index:02d}.png",
                duration_per_clip,
                force,
            )
            for index, (clip, output) in enumerate(zip(clips, outputs))
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return concat_videos(ffmpeg, outputs, destination, force)


def render_comparison_sequence(
    ffmpeg: str,
    groups: list[dict[str, Any]],
    destination: Path,
    work_dir: Path,
    duration_per_group: float,
    workers: int,
    force: bool,
) -> Path:
    group_dir = work_dir / "comparison_groups"
    overlay_dir = work_dir / "overlays"
    outputs = [group_dir / f"comparison_{index:02d}.mp4" for index in range(len(groups))]
    print(f"Rendering prompt comparison: {len(groups)} four-model grids", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as executor:
        futures = [
            executor.submit(
                render_comparison_group,
                ffmpeg,
                group,
                output,
                overlay_dir / f"comparison_{index:02d}.png",
                duration_per_group,
                force,
            )
            for index, (group, output) in enumerate(zip(groups, outputs))
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return concat_videos(ffmpeg, outputs, destination, force)


def select_command(args: argparse.Namespace) -> None:
    manifest = json.loads(FINAL_RESULTS_MANIFEST.read_text(encoding="utf-8"))
    reports = manifest["reports"]
    static_pool = load_canonical_pool(reports, "image_static", STATIC_WEIGHTS)
    dynamic_pool = load_canonical_pool(reports, "image_dynamic", DYNAMIC_WEIGHTS)
    reconstruction_pool = load_reconstruction_pool()

    selected: list[Clip] = []
    canonical_static_labels: set[str] = set()
    for model, pool in static_pool.items():
        picks = balanced_pick(pool, 2)
        selected.extend(picks)
        canonical_static_labels.update(clip.model_label for clip in picks)

    for _, pool in dynamic_pool.items():
        selected.extend(balanced_pick(pool, 2))

    # Extra reconstruction runs expand model breadth.  Skip display labels
    # already represented by the canonical static pool unless the extra is a
    # clearly named variant (Official / V2 / TI2V, etc.).
    for model, pool in reconstruction_pool.items():
        label = MODEL_LABELS.get(model, model)
        if label in canonical_static_labels and not any(
            token in label for token in ("Official", "2.3", "V2", "TI2V")
        ):
            continue
        selected.extend(balanced_pick(pool, 2))

    selected.extend(load_alayaworld_clips(4))
    selected.sort(key=lambda clip: (clip.suite, clip.model_label, -clip.score, clip.sample_id))

    comparisons = comparison_groups(static_pool)
    pool_stats = {
        "canonical_static_models": len(static_pool),
        "canonical_dynamic_models": len(dynamic_pool),
        "extra_reconstruction_models": len(reconstruction_pool),
        "canonical_static_candidates": sum(len(pool) for pool in static_pool.values()),
        "canonical_dynamic_candidates": sum(len(pool) for pool in dynamic_pool.values()),
        "extra_reconstruction_candidates": sum(len(pool) for pool in reconstruction_pool.values()),
    }
    build_dir = Path(args.build_dir).resolve()
    save_selection(build_dir, selected, comparisons, pool_stats)
    if not args.no_contact_sheets:
        make_contact_sheets(ffmpeg_path(args.ffmpeg), build_dir, selected, args.workers)


def render_command(args: argparse.Namespace) -> None:
    build_dir = Path(args.build_dir).resolve()
    selection_path = build_dir / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"selection is missing: {selection_path}; run select first")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    clips = [Clip(**item) for item in payload["selected"] if item.get("review_status") != "rejected"]
    # Protect rendering from stale selections created before the manual audit.
    clips = [
        clip
        for clip in clips
        if (clip.model, clip.sample_id) not in MANUAL_REJECTIONS
        and clip.model
        not in {
            "hunyuan-gamecraft-static-dynamic",
            "wonderworld-image-static-v3-viewable",
            "worldfm-image-static-121f-8gpu-20260702_060713",
        }
        and Path(clip.path).exists()
    ]
    if len(clips) < 72:
        raise RuntimeError(f"only {len(clips)} reviewed clips remain; refusing to build a sparse showcase")

    comparisons = payload.get("comparison_groups", [])[:4]
    all_paths = {clip.path for clip in clips}
    all_labels = {clip.model_label for clip in clips}
    for group in comparisons:
        for item in group["clips"]:
            all_paths.add(item["path"])
            all_labels.add(item["model_label"])
    model_count = len(all_labels)
    view_count = len(all_paths)
    statistic = f"{model_count} MODEL RUNS  //  {view_count} CURATED INFERENCE VIEWS"

    static = [clip for clip in clips if clip.suite == "image_static"]
    dynamic = [clip for clip in clips if clip.suite == "image_dynamic"]
    static_heroes = priority_pick(static, HERO_STATIC_PRIORITY, 12)
    dynamic_heroes = priority_pick(dynamic, HERO_DYNAMIC_PRIORITY, 10)

    allocated = {clip.path for clip in static_heroes}
    remaining_static = [clip for clip in diverse_order(static) if clip.path not in allocated]
    intro_clips = remaining_static[:12]
    allocated.update(clip.path for clip in intro_clips)
    remaining_static = [clip for clip in remaining_static if clip.path not in allocated]
    static_wall_clips = remaining_static[:24]
    allocated.update(clip.path for clip in static_wall_clips)
    finale_clips = [clip for clip in remaining_static if clip.path not in allocated]
    finale_clips = padded(finale_clips or static, 24)
    static_walls = split_chunks(static_wall_clips, 12, 2)
    dynamic_walls = split_chunks(diverse_order(dynamic), 12, 2)

    render_root = build_dir / "render"
    segment_dir = render_root / "segments"
    overlay_dir = render_root / "overlays"
    segment_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path(args.ffmpeg)
    force = bool(args.force)

    sections: list[tuple[Path, float, str]] = []

    print("Rendering opening inference wall", flush=True)
    intro = render_wall(
        ffmpeg,
        intro_clips,
        segment_dir / "01_intro.mp4",
        overlay_dir,
        columns=4,
        rows=3,
        duration=5.5,
        title="WORLD ARENA",
        subtitle="MODEL INFERENCE SHOWCASE",
        eyebrow="GENERATIVE WORLD MODELS // 2026",
        statistic=statistic,
        title_hold=5.15,
        show_labels=True,
        darken=34,
        fade_in=0.7,
        fade_out=0.18,
        force=force,
    )
    sections.append((intro, 5.5, "opening_wall"))

    static_hero_path = render_hero_sequence(
        ffmpeg,
        static_heroes,
        segment_dir / "02_static_heroes.mp4",
        render_root,
        1.65,
        args.workers,
        force,
    )
    sections.append((static_hero_path, 1.65 * len(static_heroes), "static_hero_cuts"))

    wall_titles = [
        (
            "FROM ONE FRAME",
            "TO A LIVING, EXPLORABLE WORLD",
            "CAMERA-CONTROLLED GENERATION",
        ),
        (
            "DIVERSE BY DESIGN",
            "LANDSCAPES · CITIES · INTERIORS · STYLIZED WORLDS",
            "SCENE SCALE WITHOUT LIMITS",
        ),
    ]
    for index, (wall_clips, titles) in enumerate(zip(static_walls, wall_titles), start=3):
        print(f"Rendering static inference wall {index - 2}/2", flush=True)
        path = render_wall(
            ffmpeg,
            wall_clips,
            segment_dir / f"{index:02d}_static_wall.mp4",
            overlay_dir,
            columns=4,
            rows=3,
            duration=6.25,
            title=titles[0],
            subtitle=titles[1],
            eyebrow=titles[2],
            statistic=statistic,
            title_hold=3.0,
            show_labels=True,
            force=force,
        )
        sections.append((path, 6.25, f"static_wall_{index - 2}"))

    comparison_path = render_comparison_sequence(
        ffmpeg,
        comparisons,
        segment_dir / "05_model_comparisons.mp4",
        render_root,
        3.1,
        args.workers,
        force,
    )
    sections.append((comparison_path, 3.1 * len(comparisons), "one_prompt_four_models"))

    print("Rendering dynamic inference wall 1/2", flush=True)
    dynamic_wall_a = render_wall(
        ffmpeg,
        dynamic_walls[0],
        segment_dir / "06_dynamic_wall.mp4",
        overlay_dir,
        columns=4,
        rows=3,
        duration=6.25,
        title="MOTION BECOMES MATTER",
        subtitle="ARTICULATED · FLUID · RIGID · DEFORMABLE",
        eyebrow="DYNAMIC WORLD GENERATION",
        statistic=statistic,
        title_hold=3.0,
        show_labels=True,
        force=force,
    )
    sections.append((dynamic_wall_a, 6.25, "dynamic_wall_1"))

    dynamic_hero_path = render_hero_sequence(
        ffmpeg,
        dynamic_heroes,
        segment_dir / "07_dynamic_heroes.mp4",
        render_root,
        1.7,
        args.workers,
        force,
    )
    sections.append((dynamic_hero_path, 1.7 * len(dynamic_heroes), "dynamic_hero_cuts"))

    print("Rendering dynamic inference wall 2/2", flush=True)
    dynamic_wall_b = render_wall(
        ffmpeg,
        dynamic_walls[1],
        segment_dir / "08_dynamic_wall.mp4",
        overlay_dir,
        columns=4,
        rows=3,
        duration=6.25,
        title="PHYSICS. LIFE. CONTROL.",
        subtitle="INFERENCE ACROSS EVERY MOTION REGIME",
        eyebrow="REAL WORLDS · STYLIZED WORLDS · IMPOSSIBLE WORLDS",
        statistic=statistic,
        title_hold=3.0,
        show_labels=True,
        force=force,
    )
    sections.append((dynamic_wall_b, 6.25, "dynamic_wall_2"))

    print("Rendering 24-view finale wall", flush=True)
    massive = render_wall(
        ffmpeg,
        finale_clips,
        segment_dir / "09_massive_wall.mp4",
        overlay_dir,
        columns=6,
        rows=4,
        duration=7.5,
        title=f"{model_count} MODELS. {view_count} WORLDS.",
        subtitle="ONE UNIFIED ARENA",
        eyebrow="SEE MORE · COMPARE MORE · UNDERSTAND MORE",
        statistic="GENERATE  //  EXPLORE  //  EVALUATE",
        title_hold=3.5,
        show_labels=False,
        force=force,
    )
    sections.append((massive, 7.5, "massive_24_view_wall"))

    print("Rendering closing title", flush=True)
    endcard = render_wall(
        ffmpeg,
        (static_heroes + dynamic_heroes)[:12],
        segment_dir / "10_endcard.mp4",
        overlay_dir,
        columns=4,
        rows=3,
        duration=5.5,
        title="WORLD ARENA",
        subtitle="GENERATE · EXPLORE · EVALUATE",
        eyebrow="MODEL INFERENCE SHOWCASE",
        statistic=statistic,
        title_hold=5.35,
        show_labels=False,
        darken=115,
        fade_in=0.18,
        fade_out=1.25,
        force=force,
    )
    sections.append((endcard, 5.5, "endcard"))

    video_only = render_root / "model_inference_showcase_video_only.mp4"
    print("Assembling final timeline", flush=True)
    concat_videos(ffmpeg, [path for path, _, _ in sections], video_only, force)
    expected_duration = sum(duration for _, duration, _ in sections)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_audio:
        shutil.copy2(video_only, output)
        soundtrack = None
    else:
        soundtrack = render_soundtrack(
            ffmpeg,
            expected_duration + 1.0,
            render_root / "procedural_soundtrack.m4a",
            force,
        )
        run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_only),
                "-i",
                str(soundtrack),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output),
            ],
            quiet=True,
        )

    poster = output.with_name(f"{output.stem}_poster.jpg")
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "2.8",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster),
        ],
        quiet=True,
    )
    render_manifest = {
        "schema_version": 1,
        "output": str(output),
        "poster": str(poster),
        "expected_duration_seconds": round(expected_duration, 3),
        "resolution": [1920, 1080],
        "fps": 24,
        "selected_clips": len(clips),
        "unique_inference_views": view_count,
        "display_models": model_count,
        "has_procedural_soundtrack": soundtrack is not None,
        "sections": [
            {"name": name, "duration_seconds": duration, "path": str(path)}
            for path, duration, name in sections
        ],
        "static_heroes": [clip.to_json() for clip in static_heroes],
        "dynamic_heroes": [clip.to_json() for clip in dynamic_heroes],
    }
    (build_dir / "render_manifest.json").write_text(
        json.dumps(render_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Finished {output} ({expected_duration:.1f}s, {model_count} models, "
        f"{view_count} unique inference views)",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--ffmpeg")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select", help="rank clips and build review sheets")
    select_parser.add_argument("--no-contact-sheets", action="store_true")
    select_parser.set_defaults(func=select_command)
    render_parser = subparsers.add_parser("render", help="render the approved showcase")
    render_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    render_parser.add_argument("--force", action="store_true", help="rebuild existing render segments")
    render_parser.add_argument("--no-audio", action="store_true", help="omit the procedural soundtrack")
    render_parser.set_defaults(func=render_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
