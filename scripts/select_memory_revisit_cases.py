#!/usr/bin/env python3
"""Build the deterministic 500-case Revisit-60 memory evaluation pack.

VLM-derived labels and an optional local CLIP model are allowed for case
curation only. The emitted metric protocol explicitly forbids VLMs and
LLM-as-judge during scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
STATIC_QUOTAS = {
    ("photorealistic", "indoor", "workspaces"): 25,
    ("photorealistic", "indoor", "dining_spaces"): 30,
    ("photorealistic", "indoor", "public_spaces"): 30,
    ("photorealistic", "indoor", "passageways"): 20,
    ("photorealistic", "indoor", "living_spaces"): 45,
    ("photorealistic", "outdoor", "aquatic_landscape"): 10,
    ("photorealistic", "outdoor", "city"): 45,
    ("photorealistic", "outdoor", "suburb"): 0,
    ("photorealistic", "outdoor", "terrestrial_landscape"): 50,
    ("photorealistic", "outdoor", "verdant_landscape"): 45,
    ("stylized", "indoor", "workspaces"): 10,
    ("stylized", "indoor", "dining_spaces"): 10,
    ("stylized", "indoor", "public_spaces"): 10,
    ("stylized", "indoor", "passageways"): 8,
    ("stylized", "indoor", "living_spaces"): 12,
    ("stylized", "outdoor", "aquatic_landscape"): 5,
    ("stylized", "outdoor", "city"): 10,
    ("stylized", "outdoor", "suburb"): 8,
    ("stylized", "outdoor", "terrestrial_landscape"): 17,
    ("stylized", "outdoor", "verdant_landscape"): 10,
}
DYNAMIC_QUOTAS = {
    ("photorealistic", "articulated"): 51,
    ("photorealistic", "multi_motion"): 9,
    ("stylized", "articulated"): 20,
    ("stylized", "multi_motion"): 5,
    ("stylized", "rigid"): 15,
}
MODEL_SCOPE = [
    "LingBot-World",
    "LingBot-World-v2",
    "Matrix-Game-2",
    "Matrix-Game-3",
    "Matrix-Game-3.5",
    "SANA-WM",
    "AlayaWorld",
    "DreamX-World",
    "Hunyuan-WorldPlay",
    "Infini-World",
    "ABot-World",
]
MANUAL_EXCLUSIONS = {
    "dynamic/photorealistic/images/articulated/028.jpg": "grass patch included with donkey mask",
    "dynamic/photorealistic/images/articulated/108.jpg": "background patch selected instead of antelope",
    "dynamic/photorealistic/images/articulated/128.jpg": "road selected instead of dog",
    "dynamic/photorealistic/images/articulated/202.jpg": "background patch selected instead of elephant",
    "dynamic/photorealistic/images/articulated/254.jpg": "water/background selected instead of horse",
    "dynamic/photorealistic/images/articulated/328.jpg": "grass selected instead of donkey",
    "dynamic/photorealistic/images/articulated/369.jpg": "tree/sky selected instead of turbine",
    "dynamic/photorealistic/images/articulated/389.jpg": "grass selected instead of antelope",
    "dynamic/photorealistic/images/articulated/411.jpg": "excess foreground around dog",
    "dynamic/stylized/images/articulated/144.png": "fragmented sheep mask includes ground",
    "dynamic/stylized/images/rigid/009.jpg": "roof patch selected instead of boat",
}
STABLE_TERMS = (
    "animal", "antelope", "airplane", "aircraft", "barge", "bear", "beetle",
    "bird", "boat", "bus", "butterfly", "cable car", "car", "cat", "cheetah",
    "cow", "crane", "deer", "dog", "donkey", "eagle", "elephant", "excavator",
    "ferry", "figure", "fish", "frog", "giraffe", "goat", "helicopter", "hippo",
    "horse", "hovercraft", "insect", "lion", "machine", "meerkat", "monkey",
    "owl", "panda", "parrot", "person", "rabbit", "robot", "rocket", "rover",
    "seal", "shark", "sheep", "ship", "snowplow", "scooter", "squirrel", "taxi",
    "tiger", "train", "tram", "truck", "turtle", "van", "vehicle", "warthog",
    "wheel", "wildebeest", "wind turbine", "windmill", "wolf", "zebra",
)
UNSTABLE_TERMS = (
    "cloud", "cloth", "energy", "fabric", "field", "fire", "flag", "flower",
    "fluid", "grass", "kelp", "lava", "ocean", "plant", "plume", "rain",
    "reflection", "ribbon", "river", "seed", "sky", "smoke", "spark", "stream",
    "tornado", "tree", "vortex", "water", "wave", "wheat", "whirlpool",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-weight", type=float, default=0.08)
    return parser.parse_args()


def clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def stable_label(label: str) -> bool:
    value = " ".join(label.lower().replace("_", " ").split())
    return bool(value) and not any(x in value for x in UNSTABLE_TERMS) and any(
        x in value for x in STABLE_TERMS
    )


def dct_matrix(size: int) -> np.ndarray:
    n = np.arange(size, dtype=np.float32)
    k = np.arange(size, dtype=np.float32)[:, None]
    matrix = np.cos(math.pi / size * (n + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


DCT = dct_matrix(32)


def phash(gray: Image.Image) -> int:
    pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    block = (DCT @ pixels @ DCT.T)[:8, :8]
    threshold = float(np.median(block.reshape(-1)[1:]))
    value = 0
    for bit in (block >= threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def image_stats(path: Path, keep_thumb: bool = False) -> dict[str, Any]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        thumb = image.copy()
        thumb.thumbnail((288, 288), Image.Resampling.LANCZOS)
    rgb8 = np.asarray(thumb, dtype=np.uint8)
    rgb = rgb8.astype(np.float32) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gx = np.abs(np.diff(gray, axis=1))[:-1]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    gradient = np.sqrt(gx * gx + gy * gy)
    laplace = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    histogram, _ = np.histogram(gray, bins=64, range=(0, 1))
    probability = histogram / max(float(histogram.sum()), 1.0)
    probability = probability[probability > 0]
    entropy = float(-(probability * np.log2(probability)).sum() / 6.0)
    cell_gradient: list[float] = []
    cell_luma: list[float] = []
    for row in range(4):
        for column in range(4):
            y0, y1 = row * gradient.shape[0] // 4, (row + 1) * gradient.shape[0] // 4
            x0, x1 = column * gradient.shape[1] // 4, (column + 1) * gradient.shape[1] // 4
            ly0, ly1 = row * gray.shape[0] // 4, (row + 1) * gray.shape[0] // 4
            lx0, lx1 = column * gray.shape[1] // 4, (column + 1) * gray.shape[1] // 4
            cell_gradient.append(float(np.mean(gradient[y0:y1, x0:x1])))
            cell_luma.append(float(np.mean(gray[ly0:ly1, lx0:lx1])))
    mean_luma = float(np.mean(gray))
    clipped = float(np.mean((gray <= 0.02) | (gray >= 0.98)))
    aspect = width / max(height, 1)
    layout = np.asarray(thumb.resize((4, 4), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255
    low_hist, _ = np.histogram(gray, bins=8, range=(0, 1))
    descriptor = np.concatenate([layout, 2 * low_hist / max(float(low_hist.sum()), 1.0)])
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
    result = {
        "width": width,
        "height": height,
        "aspect_ratio": aspect,
        "mean_luma": mean_luma,
        "contrast": float(np.std(gray)),
        "entropy": entropy,
        "sharpness": float(np.var(laplace)),
        "edge_density": float(np.mean(gradient >= 0.055)),
        "grid_coverage": float(np.mean(np.asarray(cell_gradient) >= 0.018)),
        "spatial_luma": float(np.std(cell_luma)),
        "exposure": clamp(math.exp(-((mean_luma - 0.5) / 0.36) ** 4) * math.exp(-3 * clipped)),
        "aspect_score": math.exp(-((math.log(max(aspect, 1e-5) / 1.60)) / 0.52) ** 2),
        "resolution_score": clamp(min(width, height) / 720),
        "_descriptor": descriptor,
        "_phash": phash(Image.fromarray(np.uint8(gray * 255), mode="L")),
    }
    if keep_thumb:
        result["_thumb"] = rgb8
    return result


def mask_stats(mask_path: Path, diagnostics: dict[str, Any], thumb: np.ndarray) -> dict[str, Any] | None:
    with Image.open(mask_path) as opened:
        mask_image = opened.convert("L")
        mask_image.thumbnail((512, 512), Image.Resampling.NEAREST)
        mask = np.asarray(mask_image, dtype=np.uint8) >= 128
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    area = float(mask.mean())
    bbox_fill = area / max(((x1 - x0 + 1) * (y1 - y0 + 1) / (width * height)), 1e-8)
    margins = [x0 / width, (width - x1 - 1) / width, y0 / height, (height - y1 - 1) / height]
    side_touches = sum(x <= 0.003 for x in margins)
    image = thumb.astype(np.float32) / 255
    small_mask = np.asarray(
        Image.fromarray(np.uint8(mask) * 255).resize((thumb.shape[1], thumb.shape[0]), Image.Resampling.NEAREST)
    ) >= 128
    gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    if small_mask.any() and (~small_mask).any():
        contrast = float(np.linalg.norm(image[small_mask].mean(0) - image[~small_mask].mean(0)) / math.sqrt(3))
    else:
        contrast = 0.0
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gradient = np.sqrt(gx * gx + gy * gy)
    inner = small_mask.copy()
    inverse = ~small_mask
    outer = inverse.copy()
    inner[1:-1, 1:-1] &= small_mask[:-2, 1:-1] & small_mask[2:, 1:-1] & small_mask[1:-1, :-2] & small_mask[1:-1, 2:]
    outer[1:-1, 1:-1] &= inverse[:-2, 1:-1] & inverse[2:, 1:-1] & inverse[1:-1, :-2] & inverse[1:-1, 2:]
    boundary = (small_mask & ~inner) | (inverse & ~outer)
    alignment = float(np.mean(gradient[boundary])) if boundary.any() else 0.0
    alignment_ratio = alignment / max(float(np.mean(gradient)), 1e-6)
    sam = clamp(float(diagnostics.get("sam_score", 0)))
    box_iou = clamp(float(diagnostics.get("box_iou", 0)))
    quality = clamp((float(diagnostics.get("mask_quality", 0)) - 1.8) / 3)
    area_score = math.exp(-0.5 * (math.log(max(area, 1e-6) / 0.075) / 0.95) ** 2)
    boundary_score = 0.5 * clamp(alignment / 0.10) + 0.5 * clamp(alignment_ratio / 2)
    score = (
        0.20 * sam + 0.12 * box_iou + 0.10 * quality
        + 0.10 * float(bool(diagnostics.get("anchor_hit", False)))
        + 0.12 * area_score + 0.08 * clamp(min(margins) / 0.065)
        + 0.08 * clamp(contrast / 0.22) + 0.10 * boundary_score
    )
    return {
        "mask_path": str(mask_path),
        "mask_area_frac": area,
        "mask_bbox_fill": bbox_fill,
        "mask_min_margin": float(min(margins)),
        "mask_side_touches": int(side_touches),
        "sam_score": sam,
        "mask_box_iou": box_iou,
        "mask_quality": float(diagnostics.get("mask_quality", 0)),
        "mask_anchor_hit": bool(diagnostics.get("anchor_hit", False)),
        "mask_border_count": int(diagnostics.get("border_count", 4)),
        "foreground_contrast": contrast,
        "mask_boundary_alignment": alignment,
        "mask_boundary_alignment_ratio": alignment_ratio,
        "mask_score": score,
        "curation_label": str(diagnostics.get("text_prompt", "")),
        "curation_vlm_confidence": clamp(float(diagnostics.get("vlm_confidence", 0))),
    }


def records(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted((root / "static").rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            parts = path.relative_to(root).parts
            if len(parts) == 5:
                _, style, domain, category, _ = parts
                result.append({"source_path": str(path), "relative_source_path": "/".join(parts), "source_type": "static", "track": "scene_revisit", "style": style, "domain": domain, "category": category})
    for path in sorted((root / "dynamic").glob("*/images/*/*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            parts = path.relative_to(root).parts
            _, style, _, category, _ = parts
            result.append({"source_path": str(path), "relative_source_path": "/".join(parts), "source_type": "dynamic", "track": "entity_reappearance", "style": style, "domain": "dynamic", "category": category, "_mask_dir": str(root / "dynamic" / style / "masks" / category / path.stem)})
    return result


def analyze(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    try:
        output.update(image_stats(Path(record["source_path"]), keep_thumb=record["source_type"] == "dynamic"))
        if record["source_type"] == "dynamic":
            directory = Path(record["_mask_dir"])
            meta_path = directory / "meta.json"
            best = None
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                masks = sorted(directory.glob("*.png"), key=lambda p: p.name)
                for index, path in enumerate(masks):
                    diagnostic = meta.get("saved_masks", [])[index] if index < len(meta.get("saved_masks", [])) else {}
                    if not stable_label(str(diagnostic.get("text_prompt", ""))):
                        continue
                    candidate = mask_stats(path, diagnostic, output["_thumb"])
                    if candidate and (best is None or candidate["mask_score"] > best["mask_score"]):
                        best = candidate
            if best:
                output.update(best)
            output.pop("_thumb", None)
        output["analysis_error"] = ""
    except Exception as exc:
        output["analysis_error"] = f"{type(exc).__name__}: {exc}"
    return output


def isolated_crop(record: dict[str, Any]) -> Image.Image:
    with Image.open(record["source_path"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    with Image.open(record["mask_path"]) as opened:
        mask = np.asarray(opened.convert("L").resize(image.size, Image.Resampling.NEAREST)) >= 128
    source = np.asarray(image)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = max(8, int(0.18 * max(x1 - x0 + 1, y1 - y0 + 1)))
    x0, x1 = max(0, x0 - pad), min(image.width, x1 + pad + 1)
    y0, y1 = max(0, y0 - pad), min(image.height, y1 + pad + 1)
    isolated = np.full_like(source, 224)
    isolated[mask] = source[mask]
    return Image.fromarray(isolated[y0:y1, x0:x1])


def clip_curate(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.clip_model is None:
        return
    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    candidates = [x for x in items if x.get("mask_path")]
    device = torch.device(args.clip_device)
    model = CLIPModel.from_pretrained(args.clip_model, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=True)
    positive = [("an illustration of " if x["style"] == "stylized" else "a photo of ") + x["curation_label"] for x in candidates]
    negative_terms = ["background scenery", "grass and ground", "a road", "trees and foliage", "sky and clouds", "an unrecognizable fragment"]
    texts = positive + ["a photo of " + x for x in negative_terms] + ["an illustration of " + x for x in negative_terms]
    with torch.inference_mode():
        tokens = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    image_features = []
    for start in range(0, len(candidates), args.clip_batch_size):
        batch = [isolated_crop(x) for x in candidates[start:start + args.clip_batch_size]]
        with torch.inference_mode():
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            features = model.get_image_features(pixel_values=pixels)
            features /= features.norm(dim=-1, keepdim=True)
        image_features.append(features.cpu())
        print(f"CLIP curation {min(start + args.clip_batch_size, len(candidates))}/{len(candidates)}", flush=True)
    image_features = torch.cat(image_features)
    text_features = text_features.cpu()
    count = len(candidates)
    for index, record in enumerate(candidates):
        similarity = float(torch.dot(image_features[index], text_features[index]))
        offset = count + (len(negative_terms) if record["style"] == "stylized" else 0)
        negative = float((image_features[index:index + 1] @ text_features[offset:offset + len(negative_terms)].T).max())
        margin = similarity - negative
        clip_score = 0.5 * clamp((similarity - 0.18) / 0.18) + 0.5 * clamp((margin + 0.10) / 0.22)
        record["clip_curation_similarity"] = similarity
        record["clip_curation_margin"] = margin
        record["clip_curation_score"] = clip_score
        weight = clamp(float(args.clip_weight))
        record["mask_score"] = (1 - weight) * record["mask_score"] + weight * clip_score


def score(items: list[dict[str, Any]]) -> None:
    keys = ["sharpness", "entropy", "contrast", "edge_density", "grid_coverage", "spatial_luma"]
    bounds = {}
    for key in keys:
        values = np.asarray([x[key] for x in items if key in x])
        low, high = np.percentile(values, [10, 90])
        bounds[key] = (float(low), float(max(high, low + 1e-8)))
    for item in items:
        if item.get("analysis_error"):
            item.update(eligible=False, exclusion_reason="analysis_error", selection_score=0.0)
            continue
        normalized = {key: clamp((item[key] - bounds[key][0]) / (bounds[key][1] - bounds[key][0])) for key in keys}
        geometry = 0.4 * item["aspect_score"] + 0.3 * item["resolution_score"] + 0.3 * item["grid_coverage"]
        scene = 0.20 * normalized["sharpness"] + 0.16 * normalized["entropy"] + 0.14 * normalized["contrast"] + 0.17 * normalized["edge_density"] + 0.11 * normalized["spatial_luma"] + 0.13 * item["exposure"] + 0.09 * geometry
        reasons = []
        if min(item["width"], item["height"]) < 384:
            reasons.append("low_resolution")
        if not 0.9 <= item["aspect_ratio"] <= 2.3:
            reasons.append("unsupported_aspect")
        if item["exposure"] < 0.18:
            reasons.append("bad_exposure")
        item["scene_selection_score"] = scene
        if item["source_type"] == "dynamic":
            if item["relative_source_path"] in MANUAL_EXCLUSIONS:
                reasons.append("manual_mask_qc_failure")
            if item["category"] in {"fluid", "deformable"}:
                reasons.append("unstable_entity_category")
            if "mask_score" not in item:
                reasons.append("missing_stable_entity_mask")
            else:
                if not 0.012 <= item["mask_area_frac"] <= 0.35:
                    reasons.append("mask_area")
                if item["sam_score"] < 0.75 or item["mask_box_iou"] < 0.35 or not item["mask_anchor_hit"]:
                    reasons.append("mask_metadata_quality")
                if max(item["mask_side_touches"], item["mask_border_count"]) >= 2:
                    reasons.append("mask_border")
                if item["style"] == "photorealistic" and item["mask_boundary_alignment"] < 0.04:
                    reasons.append("mask_boundary_alignment")
                if item["style"] == "photorealistic" and item["foreground_contrast"] < 0.025:
                    reasons.append("mask_foreground_contrast")
                if item["mask_bbox_fill"] > 0.70 and item["mask_boundary_alignment_ratio"] < 1.20:
                    reasons.append("rectangular_background_mask")
                item["selection_score"] = 0.56 * scene + 0.44 * item["mask_score"]
        else:
            item["selection_score"] = scene
        item["eligible"] = not reasons
        item["exclusion_reason"] = ";".join(reasons)


def diverse(candidates: list[dict[str, Any]], count: int, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = sorted((x for x in candidates if x.get("eligible")), key=lambda x: (-x["selection_score"], x["relative_source_path"]))
    pool = available[:max(60, count * 8)]
    chosen: list[dict[str, Any]] = []
    hashes = [x["_phash"] for x in selected]
    used = {x["relative_source_path"] for x in selected}
    while pool and len(chosen) < count:
        best, best_value = None, -1.0
        reference = chosen if chosen else selected[-24:]
        for item in pool:
            if item["relative_source_path"] in used or any(hamming(item["_phash"], x) <= 2 for x in hashes):
                continue
            diversity = 1.0 if not reference else clamp(min(float(np.linalg.norm(item["_descriptor"] - x["_descriptor"])) for x in reference) / 0.42)
            value = 0.84 * item["selection_score"] + 0.16 * diversity
            if value > best_value:
                best, best_value = item, value
        if best is None:
            break
        pool.remove(best)
        chosen.append(best)
        used.add(best["relative_source_path"])
        hashes.append(best["_phash"])
    return chosen


def quota_select(items: list[dict[str, Any]], quotas: dict[tuple[str, ...], int], fields: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    shortfall = {}
    for key, count in quotas.items():
        if not count:
            continue
        group = [x for x in items if tuple(x[field] for field in fields) == key]
        picked = diverse(group, count, selected)
        selected.extend(picked)
        if len(picked) < count:
            shortfall["/".join(key)] = count - len(picked)
    requested = sum(quotas.values())
    if len(selected) < requested:
        selected.extend(diverse(items, requested - len(selected), selected))
    if len(selected) != requested:
        raise RuntimeError(f"selected {len(selected)} of {requested}")
    return selected, shortfall


def clean_record(item: dict[str, Any], root: Path) -> dict[str, Any]:
    hidden = {"_descriptor", "_phash", "_mask_dir", "analysis_error", "eligible", "exclusion_reason", "curation_label", "curation_vlm_confidence", "clip_curation_similarity", "clip_curation_margin", "clip_curation_score"}
    output = {key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value for key, value in item.items() if key not in hidden and not key.startswith("_")}
    if item.get("mask_path"):
        output["relative_mask_path"] = Path(item["mask_path"]).relative_to(root).as_posix()
        output["curation_metadata"] = {
            "object_label": item["curation_label"],
            "vlm_confidence": round(float(item["curation_vlm_confidence"]), 6),
            "clip_similarity": round(float(item.get("clip_curation_similarity", 0)), 6),
            "clip_margin": round(float(item.get("clip_curation_margin", 0)), 6),
            "selection_only": True,
            "forbidden_metric_input": True,
        }
    output.update(
        duration_seconds=60,
        metric_uses_vlm=False,
        metric_uses_llm_as_judge=False,
        timeline_seconds={"anchor": [0, 5], "depart": [5, 20], "explore_or_occlude": [20, 36], "inverse_return": [36, 52], "revisit": [52, 60]},
    )
    if item["track"] == "scene_revisit":
        output["task_instruction"] = "Move away, explore, then return to the starting pose."
        output["metric_eligibility"] = ["revisit_return_gate", "scene_memory_f1", "memory_revisit_consistency"]
    else:
        output["task_instruction"] = "Move until the masked entity leaves view, then return and reveal it again."
        output["metric_eligibility"] = ["revisit_return_gate", "entity_reappearance_consistency", "scene_memory_f1"]
    return output


def link(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target.resolve()), str(destination))


def contact_sheet(items: list[dict[str, Any]], output: Path, prefix: str, mask_only: bool = False) -> None:
    directory = output / "contact_sheets"
    directory.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(items), 64), 1):
        canvas = Image.new("RGB", (8 * 220, 8 * 180), "white")
        draw = ImageDraw.Draw(canvas)
        for position, item in enumerate(items[start:start + 64]):
            with Image.open(item["source_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            if item.get("mask_path"):
                with Image.open(item["mask_path"]) as opened:
                    mask = np.asarray(opened.convert("L").resize(image.size, Image.Resampling.NEAREST)) >= 128
                source = np.asarray(image).copy()
                if mask_only:
                    rendered = np.full_like(source, 232)
                    rendered[mask] = source[mask]
                else:
                    rendered = source
                    rendered[mask] = (0.62 * rendered[mask] + 0.38 * np.asarray([255, 0, 0])).astype(np.uint8)
                image = Image.fromarray(rendered)
            image.thumbnail((212, 142), Image.Resampling.LANCZOS)
            column, row = position % 8, position // 8
            x = column * 220 + (220 - image.width) // 2
            y = row * 180 + 2
            canvas.paste(image, (x, y))
            draw.text((column * 220 + 4, row * 180 + 148), f"{item['case_id']} {item['category'][:13]}", fill="black")
            draw.text((column * 220 + 4, row * 180 + 163), f"score={item['selection_score']:.3f}", fill="black")
        suffix = "_mask_only" if mask_only else ""
        canvas.save(directory / f"{prefix}{suffix}_{page:02d}.jpg", quality=90, subsampling=0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main() -> None:
    args = arguments()
    root, output = args.input.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates = records(root)
    print(f"Discovered {len(candidates)} images", flush=True)
    analyzed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, item in enumerate(executor.map(analyze, candidates), 1):
            analyzed.append(item)
            if index % 250 == 0 or index == len(candidates):
                print(f"Analyzed {index}/{len(candidates)}", flush=True)
    clip_curate(analyzed, args)
    score(analyzed)
    static = [x for x in analyzed if x["source_type"] == "static"]
    dynamic = [x for x in analyzed if x["source_type"] == "dynamic"]
    selected_static, static_shortfall = quota_select(static, STATIC_QUOTAS, ("style", "domain", "category"))
    selected_dynamic, dynamic_shortfall = quota_select(dynamic, DYNAMIC_QUOTAS, ("style", "category"))
    for group, prefix in ((selected_static, "scene_revisit"), (selected_dynamic, "entity_reappearance")):
        group.sort(key=lambda x: (x["style"], x["domain"], x["category"], x["relative_source_path"]))
        for index, item in enumerate(group, 1):
            item["case_id"] = f"{prefix}_{index:04d}"
            item["selection_rank_within_track"] = index
    selected = selected_static + selected_dynamic
    public = []
    for item in selected:
        image_rel = Path("images") / item["track"] / f"{item['case_id']}{Path(item['source_path']).suffix.lower()}"
        link(Path(item["source_path"]), output / image_rel)
        row = clean_record(item, root)
        row["selected_image_path"] = image_rel.as_posix()
        if item.get("mask_path"):
            mask_rel = Path("masks") / item["track"] / f"{item['case_id']}.png"
            link(Path(item["mask_path"]), output / mask_rel)
            row["selected_mask_path"] = mask_rel.as_posix()
        public.append(row)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in public:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(output / "manifest.csv", public)
    selected_paths = {x["relative_source_path"] for x in selected}
    audit = [{"relative_source_path": x["relative_source_path"], "source_type": x["source_type"], "style": x["style"], "domain": x["domain"], "category": x["category"], "eligible": bool(x.get("eligible")), "selected": x["relative_source_path"] in selected_paths, "selection_score": round(float(x.get("selection_score", 0)), 6), "exclusion_reason": x.get("exclusion_reason", ""), "analysis_error": x.get("analysis_error", "")} for x in analyzed]
    write_csv(output / "all_candidates.csv", audit)
    counts = {name: dict(sorted(Counter(x[name] for x in selected).items())) for name in ("track", "style", "domain", "category")}
    summary = {
        "version": "memory_revisit_500_v1",
        "selection_seed": args.seed,
        "source_count": len(candidates),
        "selected_count": len(selected),
        "selected_counts": counts,
        "eligible_counts": {"static": sum(x.get("eligible", False) for x in static), "dynamic": sum(x.get("eligible", False) for x in dynamic)},
        "quota_shortfalls": {"static": static_shortfall, "dynamic": dynamic_shortfall},
        "selection_uses_existing_vlm_metadata": True,
        "selection_uses_clip_qc": args.clip_model is not None,
        "clip_model": str(args.clip_model) if args.clip_model else None,
        "metric_uses_vlm": False,
        "metric_uses_llm_as_judge": False,
        "metric_forbidden_inputs": ["curation_metadata", "VLM output", "LLM-as-judge output"],
        "model_scope": MODEL_SCOPE,
        "excluded_models": ["Warp-as-History", "minWM"],
        "manual_exclusions": MANUAL_EXCLUSIONS,
    }
    (output / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Memory Revisit 500 v1\n\n"
        "400 scene_revisit + 100 entity_reappearance cases. Every case uses a "
        "60-second anchor (0-5s), depart (5-20s), explore/occlude (20-36s), "
        "inverse return (36-52s), revisit (52-60s) protocol.\n\n"
        "Official scoring first applies a camera-pose depart-and-return hard gate. "
        "Scene memory uses point-cloud retention, hallucination, and F1 with bounded "
        "trimmed-ICP alignment. Entity memory uses masked DINOv2 appearance and mask-area "
        "consistency. DINOv2/SSIM/MSE closure scores are diagnostics. No metric uses a "
        "VLM or LLM-as-judge.\n\n"
        "Existing semantic labels and optional CLIP are used only for dataset curation; "
        "curation_metadata is forbidden as a metric input. Warp-as-History and minWM are "
        "outside the model scope.\n\n"
        "See manifest.jsonl, all_candidates.csv, selection_summary.json, and "
        "contact_sheets/.\n",
        encoding="utf-8",
    )
    contact_sheet(selected_static, output, "scene_revisit")
    contact_sheet(selected_dynamic, output, "entity_reappearance")
    contact_sheet(selected_dynamic, output, "entity_reappearance", mask_only=True)
    print(json.dumps(summary["selected_counts"], indent=2, sort_keys=True), flush=True)
    print(f"Wrote {len(selected)} cases to {output}", flush=True)


if __name__ == "__main__":
    main()
