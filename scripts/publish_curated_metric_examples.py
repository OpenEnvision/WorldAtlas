#!/usr/bin/env python3
"""Publish manually reviewed same-case metric examples into the docs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST = ROOT / "artifacts/benchmark/reports/metric_example_review_20260801/shortlist.json"
COMPARE_CACHE = ROOT / "site/.cache/compare/data-image_static.json"
BENCHMARK_MANIFEST = ROOT / "artifacts/benchmark/manifest/worldarena_manifest.jsonl"
ASSET_ROOT = ROOT / "docs/assets/metric_examples"
CURATED_ROOT = ASSET_ROOT / "curated_20260801"

# Rank is one-based within the numerically generated shortlist. Every selected
# row was subsequently reviewed as a three-timepoint input/high/low frame strip.
CURATED_RANKS = {
    "background_consistency": (8, 10, 1),
    "brightness_distribution_consistency": (6, 3, 4),
    "color_temperature_constraint": (4, 3, 12),
    "epipolar_error": (11, 3, 1),
    "epipolar_inlier_rate": (1, 6, 2),
    "feature_match_coverage": (10, 12, 3),
    "motion_accuracy": (4, 6, 2),
    "motion_magnitude": (6, 1, 9),
    "motion_smoothness": (4, 1, 8),
    "optical_flow_aepe": (3,),
    "physical_fps_error": (7, 5, 2),
    "physical_fps_intra_video_cv": (4, 6, 2),
    "physical_fps_pct_error": (9, 11, 8),
    "prompt_alignment": (9, 7, 8),
    "segment_continuity": (12, 5, 6),
    "sharpness_retention": (5, 3, 7),
}

# These retained exports contain stronger input-matched examples than the
# numerical shortlist. Keep at most three verified cases per metric.
PAIRED_SELECTIONS = {
    "perceptual_quality": (
        "image_static_photorealistic_aquatic_landscape_0079",
        "image_static_stylized_aquatic_landscape_0110",
    ),
    "style_consistency": (
        "image_static_photorealistic_dining_spaces_0150",
        "image_static_stylized_public_spaces_0146",
        "image_static_stylized_public_spaces_0059",
    ),
    "long_sequence_imaging_quality": (
        "image_static_photorealistic_public_spaces_0068",
        "image_static_stylized_terrestrial_landscape_0149",
    ),
    "rollout_aesthetic_drift": (
        "image_static_stylized_living_spaces_0080",
        "image_static_photorealistic_terrestrial_landscape_0100",
        "image_static_stylized_living_spaces_0139",
    ),
    "rollout_imaging_drift": (
        "image_static_photorealistic_aquatic_landscape_0071",
        "image_static_stylized_passageways_0014",
        "image_static_photorealistic_passageways_0081",
    ),
    "rollout_visual_score": (
        "image_static_stylized_terrestrial_landscape_0001",
        "image_static_photorealistic_suburb_0067",
        "image_static_photorealistic_city_0080",
    ),
}

# The numerical shortlist keeps only each sample's absolute high/low model.
# For these metrics that often selects a prediction whose pixels do not match
# the benchmark input. These reviewed alternatives relax only that extremum
# requirement: both selected models still have to pass the same input check.
ALTERNATE_MODEL_SELECTIONS = {
    "image_quality": (
        (
            "image_static_stylized_living_spaces_0096",
            "cosmos3-nano-image-static",
            "astra_image_static",
        ),
        (
            "image_static_photorealistic_passageways_0129",
            "cosmos3-nano-image-static",
            "astra_image_static",
        ),
        (
            "image_static_photorealistic_passageways_0140",
            "cosmos3-nano-image-static",
            "astra_image_static",
        ),
    ),
    "psnr": (
        (
            "image_static_photorealistic_workspaces_0117",
            "astra_image_static",
            "cosmos3-nano-image-static",
        ),
        (
            "image_static_photorealistic_workspaces_0024",
            "astra_image_static",
            "cosmos3-nano-image-static",
        ),
        (
            "image_static_photorealistic_living_spaces_0047",
            "astra_image_static",
            "cosmos3-nano-image-static",
        ),
    ),
    "ssim": (
        (
            "image_static_photorealistic_passageways_0076",
            "astra_image_static",
            "cosmos3-nano-image-static",
        ),
        (
            "image_static_photorealistic_dining_spaces_0032",
            "astra_image_static",
            "cosmos3-nano-image-static",
        ),
        (
            "image_static_photorealistic_verdant_landscape_0001",
            "astra_image_static",
            "cosmos3-nano-image-static",
        ),
    ),
}

FORMAL_REPORT_FILES = {
    ("hpsv3_norm", "astra_image_static"): ROOT
    / "artifacts/reports/astra_image_static_image_static_hpsv3_norm_batch8_threaded16_full_20260718/per_sample_scores.jsonl",
    ("hpsv3_norm", "cosmos3-nano-image-static"): ROOT
    / "artifacts/reports/cosmos3-nano-image-static_image_static_hpsv3_norm_batch8_inmem_threaded8_20260718/per_sample_scores.jsonl",
}

# These HPSv3 reports were generated after the frontend comparison cache, so
# reviewed rows are selected directly from the retained formal report files.
FORMAL_REPORT_SELECTIONS = {
    "hpsv3_norm": (
        (
            "image_static_photorealistic_workspaces_0020",
            "cosmos3-nano-image-static",
            "astra_image_static",
        ),
        (
            "image_static_photorealistic_verdant_landscape_0125",
            "cosmos3-nano-image-static",
            "astra_image_static",
        ),
        (
            "image_static_stylized_living_spaces_0142",
            "cosmos3-nano-image-static",
            "astra_image_static",
        ),
    ),
}

MIN_FIRST_FRAME_INLIERS = 40

DOC_FILES = {
    "background_consistency": "background_consistency.md",
    "brightness_distribution_consistency": "brightness_consistency.md",
    "color_temperature_constraint": "color_temperature_consistency.md",
    "epipolar_error": "gvgc_geometry.md",
    "epipolar_inlier_rate": "gvgc_geometry.md",
    "feature_match_coverage": "gvgc_geometry.md",
    "hpsv3_norm": "hpsv3_norm.md",
    "image_quality": "image_quality.md",
    "motion_accuracy": "motion_accuracy.md",
    "motion_magnitude": "motion_magnitude.md",
    "motion_smoothness": "motion_smoothness.md",
    "optical_flow_aepe": "optical_flow_aepe.md",
    "perceptual_quality": "perceptual_quality.md",
    "physical_fps_error": "temporal_calibration.md",
    "physical_fps_intra_video_cv": "temporal_calibration.md",
    "physical_fps_pct_error": "temporal_calibration.md",
    "prompt_alignment": "prompt_alignment.md",
    "psnr": "psnr.md",
    "segment_continuity": "segment_continuity.md",
    "sharpness_retention": "sharpness_retention.md",
    "ssim": "ssim.md",
    "style_consistency": "style_consistency.md",
    "long_sequence_imaging_quality": "long_sequence_imaging_quality.md",
    "rollout_aesthetic_drift": "rollout_aesthetic_drift.md",
    "rollout_imaging_drift": "rollout_imaging_drift.md",
    "rollout_visual_score": "long_sequence.md",
}

UNAVAILABLE_DOCS = {
    "camera_error.md": {
        "en": "The retained `artifacts/reports` snapshot does not contain two distinct predictions for the same case under one comparable Camera Error protocol. The former card compared the same video through different backends, so it has been removed rather than presented as a good/bad output pair.",
        "zh": "当前保留的 `artifacts/reports` 快照中，没有同一 case、同一 Camera Error 协议下的两条不同预测。旧卡片实际是同一视频经过不同 backend 的结果，因此已移除，不再伪装成好/差输出对照。",
    },
    "depth_accuracy.md": {
        "en": "The retained formal reports currently provide only one prediction run for this metric, so no verified same-case high/low output pair is available. Different-case extrema are intentionally not shown.",
        "zh": "当前保留的正式报告只覆盖该指标的一条 prediction run，无法组成经过核验的同-case高/低分输出对。这里刻意不再展示不同 case 的极值。",
    },
    "depth_collision.md": {
        "en": "The retained formal reports currently provide only one prediction run for this metric, so no verified same-case high/low output pair is available. Different-case extrema are intentionally not shown.",
        "zh": "当前保留的正式报告只覆盖该指标的一条 prediction run，无法组成经过核验的同-case高/低分输出对。这里刻意不再展示不同 case 的极值。",
    },
    "long_sequence_aesthetic_quality.md": {
        "en": "The retained candidates do not contain a two-sided input-verified pair for this metric. The former six-card gallery has been removed.",
        "zh": "当前保留候选中没有双侧通过输入核验的对照。原先包含六组案例的 gallery 已移除。",
    },
    "semantic_alignment.md": {
        "en": "The previous example came from readiness/smoke reports outside `artifacts/reports`. It has been removed from the formal-results gallery until a verified same-case pair exists in the retained report snapshot.",
        "zh": "旧示例来自 `artifacts/reports` 之外的 readiness/smoke 报告。它已从正式结果展示中移除，待保留报告快照中出现可核验的同-case对照后再补充。",
    },
    "reconstruction_consistency.md": {
        "en": "The retained formal snapshot contains only one completed model report for each reconstruction submetric (geometric and photometric consistency). A same-case high/low pair would therefore be fabricated, so no video comparison is shown.",
        "zh": "当前保留的正式快照中，重建一致性的两个子指标（几何与光度）都只有一个完成的模型报告。强行组成同-case高/低分对照会产生误导，因此暂不展示视频对比。",
    },
    "reprojection_error.md": {
        "en": "No comparable same-case prediction pair for this metric is retained in `artifacts/reports`. The page remains documented, but an example will be added only after two compatible formal reports are available.",
        "zh": "`artifacts/reports` 中尚未保留该指标可比较的同-case预测对。指标说明继续保留，待至少两份兼容的正式报告齐备后再加入示例。",
    },
}

NOTES = {
    "background_consistency": {
        "en": "The higher-scoring rollout keeps the lake, mountain, and shoreline coherent through the camera move; the lower-scoring rollout loses the background and ends in a collapsed texture.",
        "zh": "较高分结果在镜头移动中保持湖面、山体与岸线连贯；较低分结果逐步丢失背景，末段退化为崩坏纹理。",
    },
    "brightness_distribution_consistency": {
        "en": "The higher-scoring rollout keeps the bright coastal exposure stable; the lower-scoring rollout shifts toward a dark, murky distribution.",
        "zh": "较高分结果持续保持明亮海岸的曝光分布；较低分结果很快转成昏暗、浑浊的亮度分布。",
    },
    "color_temperature_constraint": {
        "en": "The higher-scoring rollout preserves a coherent warm palette; the lower-scoring rollout drifts into a desaturated gray appearance.",
        "zh": "较高分结果保持了连贯的暖色调；较低分结果则漂移为明显去饱和的灰色外观。",
    },
    "epipolar_error": {
        "en": "The higher-scoring camera move retains recoverable scene geometry; the lower-scoring rollout progressively darkens and loses cross-view correspondences.",
        "zh": "较高分的镜头运动仍保有可恢复的场景几何；较低分结果逐步变暗并丢失跨视角对应关系。",
    },
    "epipolar_inlier_rate": {
        "en": "The higher-scoring rollout keeps the aisle layout trackable across the orbit; the lower-scoring rollout breaks the shelf geometry into incompatible views.",
        "zh": "较高分结果在环绕中仍能追踪书架通道布局；较低分结果把书架几何拆成互不兼容的视角。",
    },
    "feature_match_coverage": {
        "en": "The higher-scoring rollout preserves trackable corridor structure across the frame; the lower-scoring rollout collapses into texture-only walls with poor spatial coverage.",
        "zh": "较高分结果在画面范围内保留了可跟踪的走廊结构；较低分结果退化为纹理墙面，可用匹配的空间覆盖明显不足。",
    },
    "hpsv3_norm": {
        "en": "The higher-scoring rollout remains a coherent, detailed sunset landscape; the lower-scoring rollout is flatter and less preference-aligned. HPSv3 scores prediction frames only.",
        "zh": "较高分结果保持了连贯且细节丰富的日落景观；较低分结果更平、更缺乏偏好一致性。HPSv3 只评价生成帧本身。",
    },
    "image_quality": {
        "en": "The higher-scoring rollout stays crisp and well rendered; the lower-scoring rollout is visibly washed out and texture-poor.",
        "zh": "较高分结果清晰、渲染完整；较低分结果明显发灰，纹理细节不足。",
    },
    "motion_accuracy": {
        "en": "The higher-scoring rollout keeps both sharks as salient moving foreground objects against a steadier seabed; the lower-scoring rollout mixes object deformation with broader scene motion.",
        "zh": "较高分结果让两条鲨鱼成为相对稳定海床上的显著运动前景；较低分结果把物体形变与更大范围的场景运动混在一起。",
    },
    "motion_magnitude": {
        "en": "The higher-scoring dog changes pose and position substantially across the clip; the lower-scoring result repeats a much narrower pose and displacement range.",
        "zh": "较高分结果中狗的姿态与位置变化更充分；较低分结果的姿态和位移范围明显更小。",
    },
    "optical_flow_aepe": {
        "en": "The higher-scoring orbit changes viewpoint while preserving a coherent atrium; the lower-scoring rollout replaces the structure, producing flow inconsistent with the intended camera motion.",
        "zh": "较高分结果在改变视角时仍保持中庭结构连贯；较低分结果替换了主体结构，光流与预期镜头运动不再一致。",
    },
    "perceptual_quality": {
        "en": "The higher-scoring rollout remains clean and photorealistic; the lower-scoring rollout contains severe tearing and reconstruction artifacts.",
        "zh": "较高分结果保持干净、写实；较低分结果存在严重撕裂与重建伪影。",
    },
    "physical_fps_error": {
        "en": "Both outputs depict the same courtyard. Judge the playback cadence: the higher-scoring output's estimated physical FPS nearly matches its container FPS, while the lower-scoring output's motion is calibrated far faster than its declared rate.",
        "zh": "两条结果展示的是同一庭院。请重点观察播放节奏：较高分结果的估计物理 FPS 与容器 FPS 几乎一致；较低分结果的运动速度则远快于其声明帧率。",
    },
    "physical_fps_intra_video_cv": {
        "en": "The higher-scoring conference-room move maintains an almost constant physical-FPS cadence; the lower-scoring output drops sharply from a fast opening to a much slower remainder. Appearance quality is not what this metric measures.",
        "zh": "较高分的会议室镜头保持近乎恒定的物理 FPS 节奏；较低分结果从开头的快速运动骤降为后段的慢速运动。该指标不评价画面外观质量。",
    },
    "physical_fps_pct_error": {
        "en": "The higher-scoring orbit has estimated physical FPS aligned with its declared FPS; the lower-scoring output moves at roughly 60 physical FPS while declaring only 13 FPS, producing a very large relative error.",
        "zh": "较高分环绕镜头的估计物理 FPS 与声明 FPS 对齐；较低分结果约以 60 物理 FPS 运动，却只声明 13 FPS，因此相对误差很大。",
    },
    "prompt_alignment": {
        "en": "The higher-scoring rollout realizes the requested attic bedroom and large window; the lower-scoring rollout drifts into outdoor and abstract content.",
        "zh": "较高分结果实现了 prompt 中的阁楼卧室与大窗；较低分结果则漂移到室外和抽象内容。",
    },
    "psnr": {
        "en": "The higher-scoring rollout stays closer to the reference pixels over the evaluated frames; the lower-scoring rollout accumulates larger color and structure differences. PSNR measures reference fidelity, not standalone aesthetics.",
        "zh": "较高分结果在被评估帧上更接近参考图像素；较低分结果累积了更大的颜色与结构差异。PSNR 衡量参考保真度，不等同于独立审美质量。",
    },
    "segment_continuity": {
        "en": "The higher-scoring beach rollout stays within one continuous scene; the lower-scoring rollout contains a detected hard cut from the beach hut to an unrelated harbor scene.",
        "zh": "较高分海滩结果始终处于同一连续场景；较低分结果从沙滩棚突然硬切到无关的港口场景，并被检测为一次 cut。",
    },
    "sharpness_retention": {
        "en": "The higher-scoring rollout retains crisp edges and fine structure; the lower-scoring rollout progressively blurs and dissolves the city scene.",
        "zh": "较高分结果保留了清晰边缘和细部结构；较低分结果中的城市画面逐步模糊、溶解。",
    },
    "ssim": {
        "en": "The higher-scoring rollout retains the input room's spatial layout and major structures; the lower-scoring rollout drifts to generic, differently arranged interiors. SSIM emphasizes structural fidelity to the reference.",
        "zh": "较高分结果保留了输入房间的空间布局和主要结构；较低分结果漂移成布局不同的通用室内场景。SSIM 更强调对参考图的结构保真度。",
    },
    "style_consistency": {
        "en": "The higher-scoring rollout stays within a coherent photoreal geometric-interior style; the lower-scoring rollout switches to an unrelated reflective striped appearance.",
        "zh": "较高分结果保持连贯的写实几何室内风格；较低分结果则切换成无关的反光条纹外观。",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_key(path: str) -> str:
    marker = "artifacts/predictions/"
    return path.split(marker, 1)[1] if marker in path else path


def remap_cpfs(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if path.startswith("/mnt/cpfs/"):
        candidate = Path("/mnt/cpfsB") / path.removeprefix("/mnt/cpfs/")
    return candidate


def old_assets() -> dict[str, str]:
    video_map: dict[str, str] = {}
    for row in load_json(ASSET_ROOT / "paired_manifest.json"):
        if row.get("prediction_path") and row.get("asset_path"):
            video_map[prediction_key(str(row["prediction_path"]))] = str(row["asset_path"])
    return video_map


def relative_asset(path: Path) -> str:
    return path.relative_to(ROOT / "docs").as_posix()


def publish_input(row: dict[str, Any]) -> str:
    source_path = remap_cpfs(str(row["source_path"]))
    source = Image.open(source_path).convert("RGB")
    source.thumbnail((960, 720), Image.Resampling.LANCZOS)
    digest = sha256_file(source_path)[:12]
    output = CURATED_ROOT / f"{digest}__{row['sample_id']}__input.jpg"
    if not output.is_file():
        source.save(output, quality=90, optimize=True)
    return relative_asset(output)


def transcode_video(source: Path, output: Path) -> None:
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    temporary.unlink(missing_ok=True)

    try:
        with av.open(str(source)) as container:
            source_stream = container.streams.video[0]
            source_rate = float(source_stream.average_rate or 12)
            target_rate = max(1, round(min(source_rate, 12.0)))
            target_time_base = Fraction(1, target_rate)
            scale = min(1.0, 640 / source_stream.width, 360 / source_stream.height)
            width = max(2, math.floor(source_stream.width * scale / 2) * 2)
            height = max(2, math.floor(source_stream.height * scale / 2) * 2)

            with av.open(
                str(temporary),
                mode="w",
                options={"movflags": "+faststart"},
            ) as encoded:
                target_stream = encoded.add_stream("libx264", rate=target_rate)
                target_stream.width = width
                target_stream.height = height
                target_stream.pix_fmt = "yuv420p"
                target_stream.time_base = target_time_base
                target_stream.options = {"crf": "25", "preset": "medium"}

                next_time = 0.0
                output_index = 0
                for frame in container.decode(source_stream):
                    frame_time = float(frame.time or 0.0)
                    if frame_time + 1e-6 < next_time:
                        continue
                    next_time += 1.0 / target_rate
                    resized = frame.reformat(width=width, height=height, format="yuv420p")
                    resized.pts = output_index
                    resized.time_base = target_time_base
                    output_index += 1
                    for packet in target_stream.encode(resized):
                        encoded.mux(packet)
                for packet in target_stream.encode():
                    encoded.mux(packet)

        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def video_requires_transcode(path: Path) -> bool:
    if not path.is_file():
        return True

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            average_rate = float(stream.average_rate or 0)
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else 0.0
            )
    except (av.error.FFmpegError, OSError, IndexError, TypeError, ValueError):
        return True

    return not (0 < average_rate <= 60 and duration > 0)


def publish_video(row: dict[str, Any], side: str, video_map: dict[str, str]) -> str:
    source_value = str(row[side]["prediction_path"])
    key = prediction_key(source_value)
    existing = video_map.get(key)
    if existing and (ROOT / existing).is_file():
        return str(Path(existing).relative_to("docs"))
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    output = CURATED_ROOT / f"{digest}__{row['sample_id']}__{side}.mp4"
    if video_requires_transcode(output):
        local_prediction = ROOT / "artifacts/predictions" / key
        source = local_prediction if local_prediction.is_file() else remap_cpfs(source_value)
        transcode_video(source, output)
    return relative_asset(output)


def paired_selection(metric: str, sample_id: str) -> dict[str, Any]:
    candidates = [
        row
        for row in load_json(ASSET_ROOT / "paired_manifest.json")
        if row.get("metric") == metric and row.get("sample_id") == sample_id
    ]
    if len(candidates) != 2:
        raise ValueError(f"expected two paired rows for {metric}/{sample_id}, got {len(candidates)}")
    by_side = {str(row["side"]): row for row in candidates}
    if set(by_side) != {"good", "bad"}:
        raise ValueError(f"unexpected paired sides for {metric}/{sample_id}: {sorted(by_side)}")
    good = by_side["good"]
    bad = by_side["bad"]
    return {
        "metric": metric,
        "suite": good["suite"],
        "sample_id": sample_id,
        "source_path": good["source_path"],
        "prompt": good["prompt"],
        "prompt_source": good["prompt_source"],
        "style": good.get("style"),
        "environment": good.get("environment"),
        "scene": good.get("scene"),
        "motion_category": good.get("motion_category"),
        "gap": float(good.get("pair_gap") or 0.0),
        "selection_source": "paired_manifest",
        "high": {
            "prediction_path": good["prediction_path"],
            "normalized": float(good["normalized"]),
            "raw": good.get("raw"),
            "details": {},
        },
        "low": {
            "prediction_path": bad["prediction_path"],
            "normalized": float(bad["normalized"]),
            "raw": bad.get("raw"),
            "details": {},
        },
    }


def load_benchmark_manifest() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with BENCHMARK_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def alternate_model_selection(
    metric: str,
    sample_id: str,
    high_model_name: str,
    low_model_name: str,
    comparison_cache: dict[str, Any],
    benchmark_manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparison = comparison_cache[sample_id]
    benchmark = benchmark_manifest[sample_id]
    models = {str(model["runName"]): model for model in comparison["models"]}

    def side(model_name: str) -> dict[str, Any]:
        model = models[model_name]
        if metric not in model.get("scores", {}):
            raise ValueError(f"missing {metric} score for {model_name}/{sample_id}")
        prediction_path = ROOT / "artifacts/predictions" / model_name / f"{sample_id}.mp4"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        return {
            "suite": str(comparison["suite"]),
            "metric": metric,
            "model": model_name,
            "sample_id": sample_id,
            "normalized": float(model["scores"][metric]),
            "raw": None,
            "details": {},
            "backend": "",
            "prediction_path": str(prediction_path),
            "source_path": str(benchmark["path"]),
            "style": benchmark.get("style"),
            "environment": benchmark.get("environment"),
            "scene": benchmark.get("scene"),
            "motion_category": benchmark.get("motion_category"),
            "report_dir": "artifacts/reports (frontend comparison cache)",
            "report_generated_at": "2026-07-17T22:53:22.381Z",
            "model_aggregate_score": None,
        }

    high = side(high_model_name)
    low = side(low_model_name)
    if high["normalized"] <= low["normalized"]:
        raise ValueError(
            f"expected high > low for {metric}/{sample_id}, got "
            f"{high['normalized']} <= {low['normalized']}"
        )
    prompt = benchmark.get("prompt_target") or benchmark.get("prompt_current") or ""
    prompt_source = "prompt_target" if benchmark.get("prompt_target") else "prompt_current"
    return {
        "metric": metric,
        "suite": str(comparison["suite"]),
        "sample_id": sample_id,
        "source_path": str(benchmark["path"]),
        "prompt": prompt,
        "prompt_source": prompt_source,
        "style": benchmark.get("style"),
        "environment": benchmark.get("environment"),
        "scene": benchmark.get("scene"),
        "motion_category": benchmark.get("motion_category"),
        "gap": high["normalized"] - low["normalized"],
        "selection_source": "reviewed_alternate_model_pair",
        "high": high,
        "low": low,
    }


def load_formal_report(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def formal_report_selection(
    metric: str,
    sample_id: str,
    high_model_name: str,
    low_model_name: str,
    report_rows: dict[tuple[str, str], dict[str, dict[str, Any]]],
    benchmark_manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark = benchmark_manifest[sample_id]

    def side(model_name: str) -> dict[str, Any]:
        report_path = FORMAL_REPORT_FILES[(metric, model_name)]
        report_row = report_rows[(metric, model_name)][sample_id]
        result = report_row.get("metrics", {}).get(metric, {})
        if result.get("error") is not None or result.get("normalized") is None:
            raise ValueError(f"invalid {metric} result for {model_name}/{sample_id}")
        prediction_path = remap_cpfs(str(report_row["prediction_path"]))
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        return {
            "suite": str(report_row["suite"]),
            "metric": metric,
            "model": model_name,
            "sample_id": sample_id,
            "normalized": float(result["normalized"]),
            "raw": result.get("raw"),
            "details": result.get("details") or {},
            "backend": result.get("backend") or "",
            "prediction_path": str(prediction_path),
            "source_path": str(report_row["path"]),
            "style": report_row.get("style"),
            "environment": report_row.get("environment"),
            "scene": report_row.get("scene"),
            "motion_category": report_row.get("motion_category"),
            "report_dir": report_path.parent.relative_to(ROOT).as_posix(),
            "report_generated_at": "2026-07-18",
            "model_aggregate_score": None,
        }

    high = side(high_model_name)
    low = side(low_model_name)
    if remap_cpfs(str(high["source_path"])) != remap_cpfs(str(low["source_path"])):
        raise ValueError(f"different source paths for {metric}/{sample_id}")
    if high["normalized"] <= low["normalized"]:
        raise ValueError(
            f"expected high > low for {metric}/{sample_id}, got "
            f"{high['normalized']} <= {low['normalized']}"
        )
    prompt = benchmark.get("prompt_target") or benchmark.get("prompt_current") or ""
    prompt_source = "prompt_target" if benchmark.get("prompt_target") else "prompt_current"
    return {
        "metric": metric,
        "suite": str(high["suite"]),
        "sample_id": sample_id,
        "source_path": str(high["source_path"]),
        "prompt": prompt,
        "prompt_source": prompt_source,
        "style": benchmark.get("style"),
        "environment": benchmark.get("environment"),
        "scene": benchmark.get("scene"),
        "motion_category": benchmark.get("motion_category"),
        "gap": high["normalized"] - low["normalized"],
        "selection_source": "reviewed_formal_report_pair",
        "high": high,
        "low": low,
    }


def select_rows() -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_json(SHORTLIST):
        grouped[row["metric"]].append(row)
    selected = []
    for metric, ranks in CURATED_RANKS.items():
        for rank in ranks:
            row = grouped[metric][rank - 1]
            row["selection_source"] = "review_shortlist"
            row["selection_rank"] = rank
            selected.append(row)
    for metric, sample_ids in PAIRED_SELECTIONS.items():
        selected.extend(paired_selection(metric, sample_id) for sample_id in sample_ids)
    comparison_cache = load_json(COMPARE_CACHE)
    benchmark_manifest = load_benchmark_manifest()
    for metric, selections in ALTERNATE_MODEL_SELECTIONS.items():
        selected.extend(
            alternate_model_selection(
                metric,
                sample_id,
                high_model_name,
                low_model_name,
                comparison_cache,
                benchmark_manifest,
            )
            for sample_id, high_model_name, low_model_name in selections
        )
    report_rows = {
        key: load_formal_report(path) for key, path in FORMAL_REPORT_FILES.items()
    }
    for metric, selections in FORMAL_REPORT_SELECTIONS.items():
        selected.extend(
            formal_report_selection(
                metric,
                sample_id,
                high_model_name,
                low_model_name,
                report_rows,
                benchmark_manifest,
            )
            for sample_id, high_model_name, low_model_name in selections
        )
    return selected


def first_frame(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
    return cv2.cvtColor(np.asarray(frame.to_image().convert("RGB")), cv2.COLOR_RGB2GRAY)


def geometric_inliers(reference: np.ndarray, candidate: np.ndarray) -> int:
    detector = cv2.SIFT_create(nfeatures=2200)
    reference_points, reference_descriptors = detector.detectAndCompute(reference, None)
    candidate_points, candidate_descriptors = detector.detectAndCompute(candidate, None)
    if reference_descriptors is None or candidate_descriptors is None:
        return 0
    matches = cv2.BFMatcher().knnMatch(reference_descriptors, candidate_descriptors, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    if len(good) < 4:
        return 0
    source = np.float32([reference_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    target = np.float32([candidate_points[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def verify_input_match(row: dict[str, Any]) -> None:
    input_path = ROOT / "docs" / row["input_asset"]
    reference = cv2.cvtColor(np.asarray(Image.open(input_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    high_inliers = geometric_inliers(reference, first_frame(ROOT / "docs" / row["high_asset"]))
    low_inliers = geometric_inliers(reference, first_frame(ROOT / "docs" / row["low_asset"]))
    if min(high_inliers, low_inliers) < MIN_FIRST_FRAME_INLIERS:
        raise ValueError(
            f"input verification failed for {row['metric']}/{row['sample_id']}: "
            f"high={high_inliers}, low={low_inliers}, required={MIN_FIRST_FRAME_INLIERS}"
        )
    source_path = remap_cpfs(str(row["source_path"]))
    row["input_verification"] = {
        "method": "SIFT first-frame matching with RANSAC homography",
        "minimum_required_inliers": MIN_FIRST_FRAME_INLIERS,
        "high_inliers": high_inliers,
        "low_inliers": low_inliers,
        "source_sha256": sha256_file(source_path),
        "published_input_sha256": sha256_file(input_path),
        "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest(),
    }


def asset_src(asset: str) -> str:
    return "../../../" + asset


def render_diagnostic(row: dict[str, Any], side: str, lang: str) -> str:
    metric = row["metric"]
    result = row[side]
    details = result.get("details") or {}
    zh = lang == "zh"
    if metric == "background_consistency":
        value = details.get("mean_pairwise_cosine", result.get("raw"))
        label = "平均相邻帧相似度" if zh else "Mean adjacent-frame similarity"
        return f"{label}: {value:.4f}"
    if metric == "segment_continuity":
        cuts = int(details.get("cut_count", 0))
        label = "检测到的硬切次数" if zh else "Detected hard cuts"
        return f"{label}: {cuts}"
    if metric == "physical_fps_error":
        estimated = float(details["avg_phyfps"])
        declared = float(details["meta_fps"])
        error = float(result["raw"])
        if zh:
            return f"估计物理 FPS {estimated:.1f} · 声明 FPS {declared:.1f} · 绝对误差 {error:.1f}"
        return f"Estimated physical FPS {estimated:.1f} · declared FPS {declared:.1f} · absolute error {error:.1f}"
    if metric == "physical_fps_pct_error":
        estimated = float(details["avg_phyfps"])
        declared = float(details["meta_fps"])
        error_pct = float(result["raw"]) * 100
        if zh:
            return f"估计物理 FPS {estimated:.1f} · 声明 FPS {declared:.1f} · 相对误差 {error_pct:.2f}%"
        return f"Estimated physical FPS {estimated:.1f} · declared FPS {declared:.1f} · relative error {error_pct:.2f}%"
    if metric == "physical_fps_intra_video_cv":
        segments = [float(value) for value in details.get("segment_phyfps", [])]
        low = min(segments)
        high = max(segments)
        cv = float(result["raw"])
        if zh:
            return f"段间物理 FPS {low:.1f}–{high:.1f} · 变异系数 {cv:.3f}"
        return f"Segment physical FPS {low:.1f}–{high:.1f} · coefficient of variation {cv:.3f}"
    return ""


def render_case(row: dict[str, Any], lang: str) -> str:
    zh = lang == "zh"
    metric = row["metric"]
    scene_bits = [row.get("style"), row.get("environment"), row.get("scene"), row.get("motion_category")]
    scene = " / ".join(str(bit) for bit in scene_bits if bit)
    high_label = "较高指标分" if zh else "Higher metric score"
    low_label = "较低指标分" if zh else "Lower metric score"
    input_label = "输入图像" if zh else "Input image"
    prompt_label = "基准 prompt target" if zh else "Benchmark prompt target"
    reviewed = "输入已核验 · 同一 benchmark case" if zh else "Input verified · same benchmark case"
    suite_label = "套件" if zh else "Suite"
    scene_label = "场景" if zh else "Scene"
    score_label = "分数" if zh else "Score"
    note = (
        "两条视频的首帧都已通过与展示输入的几何匹配核验；高低标签只比较当前 metric。"
        if zh
        else "Both first frames passed geometric matching against the displayed input; labels compare this metric only."
    )
    high_diagnostic = render_diagnostic(row, "high", lang)
    low_diagnostic = render_diagnostic(row, "low", lang)
    high_diagnostic_html = f"<span>{high_diagnostic}</span>" if high_diagnostic else ""
    low_diagnostic_html = f"<span>{low_diagnostic}</span>" if low_diagnostic else ""
    return f'''<article class="metric-case-exhibit metric-case-exhibit--curated">
  <div class="metric-case-context">
    <figure class="metric-input-card">
      <img src="{asset_src(row['input_asset'])}" alt="{input_label}: {row['sample_id']}" loading="lazy" decoding="async">
      <figcaption>{input_label}</figcaption>
    </figure>
    <div class="metric-case-copy">
      <div class="metric-case-kicker">{reviewed} · gap {row['gap']:.4f}</div>
      <h4><code>{row['sample_id']}</code></h4>
      <div class="metric-case-meta"><span>{suite_label}: {row['suite']}</span><span>{scene_label}: {scene}</span></div>
      <div class="metric-case-prompt">
        <span>{prompt_label}</span>
        <p>{row['prompt']}</p>
        <small>prompt source: {row['prompt_source']}</small>
      </div>
      <p class="metric-case-note">{note}</p>
    </div>
  </div>
  <div class="metric-output-grid">
    <figure class="metric-output-card metric-output-card--good">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="{high_label}: {row['sample_id']}">
          <source src="{asset_src(row['high_asset'])}" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--good">{high_label} · {score_label} {row['high']['normalized']:.4f}</div>
      </div>
      <figcaption class="metric-output-caption"><strong>{high_label}</strong><span>{metric}</span>{high_diagnostic_html}</figcaption>
    </figure>
    <figure class="metric-output-card metric-output-card--bad">
      <div class="metric-output-media-wrap">
        <video class="metric-output-video" controls muted loop playsinline preload="metadata" aria-label="{low_label}: {row['sample_id']}">
          <source src="{asset_src(row['low_asset'])}" type="video/mp4">
        </video>
        <div class="metric-score-pill metric-score-pill--bad">{low_label} · {score_label} {row['low']['normalized']:.4f}</div>
      </div>
      <figcaption class="metric-output-caption"><strong>{low_label}</strong><span>{metric}</span>{low_diagnostic_html}</figcaption>
    </figure>
  </div>
</article>'''


def render_gallery(rows: list[dict[str, Any]], lang: str) -> str:
    zh = lang == "zh"
    title = "## 已核验的同输入案例" if zh else "## Verified Same-Input Examples"
    intro = (
        "每个 metric 最多展示三个 case。每组中的两条视频必须属于同一 benchmark sample，且两侧首帧都通过与该组展示输入的 SIFT + RANSAC 几何匹配；不通过的候选不会发布。标签只表示当前 metric 的相对高低，不代表模型整体优劣。"
        if zh
        else "Up to three cases are shown per metric. Within each case, both videos must belong to the same benchmark sample, and both first frames must pass SIFT + RANSAC geometric matching against that case's displayed input. Candidates that fail are not published. Labels compare this metric only, not overall model quality."
    )
    return (
        "<!-- metric-examples:start -->\n"
        f"{title}\n\n{intro}\n\n"
        "Audit manifest: `docs/assets/metric_examples/curated_same_case_manifest.json`.\n\n"
        '<div class="metric-case-stack">\n'
        + "\n".join(render_case(row, lang) for row in rows)
        + "\n</div>\n<!-- metric-examples:end -->"
    )


def replace_gallery(path: Path, gallery: str) -> None:
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- metric-examples:start -->[\s\S]*?<!-- metric-examples:end -->")
    if pattern.search(content):
        content = pattern.sub(gallery, content, count=1)
    else:
        content = content.rstrip() + "\n\n" + gallery + "\n"
    path.write_text(content, encoding="utf-8")


def unavailable_gallery(reason: str, lang: str) -> str:
    title = "## 暂无可核验的同 Case 对比" if lang == "zh" else "## No Verified Same-Case Pair Yet"
    return (
        "<!-- metric-examples:start -->\n"
        f"{title}\n\n"
        f"> {reason}\n"
        "<!-- metric-examples:end -->"
    )


def reduce_representative_gallery(
    path: Path,
    *,
    start_marker: str,
    end_marker: str,
    metric_name: str,
    manifest_name: str,
    lang: str,
    keep_all_cases: bool = False,
) -> None:
    manifest_rows = list(csv.DictReader((ASSET_ROOT / manifest_name).open(encoding="utf-8")))
    chosen = manifest_rows if keep_all_cases else [row for row in manifest_rows if str(row.get("case_rank")) == "3"]
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in chosen:
        grouped[str(row.get("case_rank"))].append(row)
    for rank, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"expected two case-rank {rank} rows in {manifest_name}, got {len(rows)}")
        input_path = ASSET_ROOT / "inputs" / rows[0]["input_asset"]
        reference = cv2.cvtColor(np.asarray(Image.open(input_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
        inliers = [geometric_inliers(reference, first_frame(ASSET_ROOT / row["video_asset"])) for row in rows]
        if min(inliers) < MIN_FIRST_FRAME_INLIERS:
            raise ValueError(
                f"representative input verification failed for {metric_name} case {rank}: "
                f"sides={inliers}, required={MIN_FIRST_FRAME_INLIERS}"
            )

    content = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start_marker) + r"[\s\S]*?" + re.escape(end_marker))
    match = pattern.search(content)
    if not match:
        raise ValueError(f"representative markers missing in {path}")
    articles = re.findall(r'<article class="metric-case-exhibit[\s\S]*?</article>', match.group(0))
    if not articles:
        raise ValueError(f"representative article missing in {path}")
    if keep_all_cases:
        if len(articles) != len(grouped):
            raise ValueError(
                f"expected {len(grouped)} representative articles in {path}, got {len(articles)}"
            )
        return
    article = re.sub(r"Case 3|案例 3", "核验案例" if lang == "zh" else "Verified case", articles[-1], count=1)
    if lang == "zh":
        heading = f"## 已核验的同输入 {metric_name} 案例"
        intro = (
            "本页只保留一个案例。两条视频的首帧都已通过与展示输入的几何匹配；"
            "分数仅用于解释当前 metric，不代表模型整体排名。"
        )
        manifest_label = "机器可读审计清单"
    else:
        heading = f"## Verified Same-Input {metric_name} Example"
        intro = (
            "This page keeps one case only. Both video first frames pass geometric matching against "
            "the displayed input; scores illustrate this metric only, not overall model ranking."
        )
        manifest_label = "Machine-readable audit manifest"
    replacement = (
        f"{start_marker}\n{heading}\n\n{intro}\n\n"
        f"[{manifest_label}](../../../assets/metric_examples/{manifest_name}).\n\n"
        '<div class="metric-case-stack">\n'
        f"{article}\n"
        "</div>\n"
        f"{end_marker}"
    )
    path.write_text(pattern.sub(replacement, content, count=1), encoding="utf-8")


def main() -> None:
    CURATED_ROOT.mkdir(parents=True, exist_ok=True)
    video_map = old_assets()
    selected = select_rows()
    for row in selected:
        row["input_asset"] = publish_input(row)
        row["high_asset"] = publish_video(row, "high", video_map)
        row["low_asset"] = publish_video(row, "low", video_map)
        verify_input_match(row)

    audit = {
        "version": 5,
        "reviewed_at": "2026-08-08",
        "source_reports_root": "artifacts/reports",
        "sources": [
            {
                "path": "site/.cache/compare",
                "description": "same-case comparison cache derived from retained formal reports",
                "generated_at": "2026-07-17T22:53:22.381Z",
            },
            {
                "path": "artifacts/reports",
                "description": "retained formal reports, including HPSv3, background, segment continuity, and physical-FPS metrics",
            },
        ],
        "selection": "up to three cases per metric; shortlist/manual selection, including reviewed non-extreme model pairs, followed by mandatory first-frame geometric input verification",
        "same_case_fields": [
            "suite",
            "sample_id",
            "input_verification.source_sha256",
            "input_verification.prompt_sha256",
        ],
        "visible_model_names": False,
        "examples": selected,
    }
    (ASSET_ROOT / "curated_same_case_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_file: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_file[DOC_FILES[row["metric"]]].append(row)
    for lang in ("en", "zh"):
        for filename, rows in by_file.items():
            replace_gallery(ROOT / f"docs/{lang}/benchmark/metrics/{filename}", render_gallery(rows, lang))
        for filename, reasons in UNAVAILABLE_DOCS.items():
            replace_gallery(
                ROOT / f"docs/{lang}/benchmark/metrics/{filename}",
                unavailable_gallery(reasons[lang], lang),
            )
        reduce_representative_gallery(
            ROOT / f"docs/{lang}/benchmark/metrics/temporal_flickering.md",
            start_marker="<!-- temporal-flickering-representative-cases:start -->",
            end_marker="<!-- temporal-flickering-representative-cases:end -->",
            metric_name="Temporal Flickering",
            manifest_name="temporal_flickering_representative_manifest.csv",
            lang=lang,
        )
        reduce_representative_gallery(
            ROOT / f"docs/{lang}/benchmark/metrics/met3r_consistency.md",
            start_marker="<!-- met3r-representative-cases:start -->",
            end_marker="<!-- met3r-representative-cases:end -->",
            metric_name="MEt3R",
            manifest_name="met3r_representative_manifest.csv",
            lang=lang,
            keep_all_cases=True,
        )

    print(
        json.dumps(
            {
                "published_examples": len(selected),
                "updated_doc_files": (len(by_file) + len(UNAVAILABLE_DOCS) + 2) * 2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
