#!/usr/bin/env python3
"""Publish manually reviewed same-case metric examples into the docs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST = ROOT / "artifacts/benchmark/reports/metric_example_review_20260801/shortlist.json"
ASSET_ROOT = ROOT / "docs/assets/metric_examples"
CURATED_ROOT = ASSET_ROOT / "curated_20260801"

# Rank is one-based within the numerically generated shortlist. Every selected
# row was subsequently reviewed as a three-timepoint input/high/low frame strip.
CURATED_RANKS = {
    "background_consistency": 1,
    "brightness_distribution_consistency": 1,
    "color_temperature_constraint": 1,
    "epipolar_error": 1,
    "epipolar_inlier_rate": 1,
    "feature_match_coverage": 1,
    "hpsv3_norm": 1,
    "image_quality": 1,
    "motion_accuracy": 2,
    "motion_magnitude": 1,
    "optical_flow_aepe": 1,
    "perceptual_quality": 1,
    "physical_fps_error": 3,
    "physical_fps_intra_video_cv": 2,
    "physical_fps_pct_error": 3,
    "prompt_alignment": 1,
    "psnr": 2,
    "segment_continuity": 2,
    "sharpness_retention": 1,
    "ssim": 2,
    "style_consistency": 3,
}

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
        "en": "The higher-scoring rollout preserves the input workspace's structure and appearance across the move; the lower-scoring rollout replaces it with unrelated close-ups. PSNR measures reference fidelity, not standalone aesthetics.",
        "zh": "较高分结果在镜头移动中保留了输入工作区的结构与外观；较低分结果则切换成无关的局部特写。PSNR 衡量参考保真度，不等同于独立审美质量。",
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


def old_assets() -> tuple[dict[str, str], dict[str, str]]:
    video_map: dict[str, str] = {}
    input_map: dict[str, str] = {}
    for row in load_json(ASSET_ROOT / "paired_manifest.json"):
        if row.get("prediction_path") and row.get("asset_path"):
            video_map[prediction_key(str(row["prediction_path"]))] = str(row["asset_path"])
        if row.get("sample_id") and row.get("input_asset_path"):
            input_map[str(row["sample_id"])] = str(row["input_asset_path"])
    return video_map, input_map


def relative_asset(path: Path) -> str:
    return path.relative_to(ROOT / "docs").as_posix()


def publish_input(row: dict[str, Any], input_map: dict[str, str]) -> str:
    existing = input_map.get(row["sample_id"])
    if existing and (ROOT / existing).is_file():
        return str(Path(existing).relative_to("docs"))
    source = Image.open(remap_cpfs(row["source_path"])).convert("RGB")
    source.thumbnail((960, 720), Image.Resampling.LANCZOS)
    digest = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:12]
    output = CURATED_ROOT / f"{digest}__{row['sample_id']}__input.jpg"
    source.save(output, quality=90, optimize=True)
    return relative_asset(output)


def transcode_video(source: Path, output: Path) -> None:
    container = av.open(str(source))
    source_stream = container.streams.video[0]
    source_rate = float(source_stream.average_rate or 12)
    target_rate = min(source_rate, 12.0)
    scale = min(1.0, 640 / source_stream.width, 360 / source_stream.height)
    width = max(2, math.floor(source_stream.width * scale / 2) * 2)
    height = max(2, math.floor(source_stream.height * scale / 2) * 2)

    encoded = av.open(str(output), mode="w")
    target_stream = encoded.add_stream("libx264", rate=round(target_rate))
    target_stream.width = width
    target_stream.height = height
    target_stream.pix_fmt = "yuv420p"
    target_stream.options = {"crf": "25", "preset": "medium", "movflags": "+faststart"}

    next_time = 0.0
    for frame in container.decode(source_stream):
        frame_time = float(frame.time or 0.0)
        if frame_time + 1e-6 < next_time:
            continue
        next_time += 1.0 / target_rate
        resized = frame.reformat(width=width, height=height, format="yuv420p")
        resized.pts = None
        for packet in target_stream.encode(resized):
            encoded.mux(packet)
    for packet in target_stream.encode():
        encoded.mux(packet)
    encoded.close()
    container.close()


def publish_video(row: dict[str, Any], side: str, video_map: dict[str, str]) -> str:
    source_value = str(row[side]["prediction_path"])
    key = prediction_key(source_value)
    existing = video_map.get(key)
    if existing and (ROOT / existing).is_file():
        return str(Path(existing).relative_to("docs"))
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    output = CURATED_ROOT / f"{digest}__{row['sample_id']}__{side}.mp4"
    if not output.is_file():
        transcode_video(remap_cpfs(source_value), output)
    return relative_asset(output)


def select_rows() -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_json(SHORTLIST):
        grouped[row["metric"]].append(row)
    return [grouped[metric][rank - 1] for metric, rank in CURATED_RANKS.items()]


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
    prompt_label = "实际生成 prompt" if zh else "Exact generation prompt"
    reviewed = "人工复核 · 同一 case、输入与 prompt" if zh else "Human reviewed · same case, input and prompt"
    suite_label = "套件" if zh else "Suite"
    scene_label = "场景" if zh else "Scene"
    score_label = "分数" if zh else "Score"
    note = NOTES[metric][lang]
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
    title = "## 人工精选的同 Case 对比" if zh else "## Human-reviewed Same-Case Example"
    intro = (
        "下列案例来自 `artifacts/reports` 生成的同-case比较索引。输入图、sample ID 与实际生成 prompt 完全一致；候选先按分位数和分差筛选，再逐帧人工复核。标签只表示该 metric 的相对高低，不代表模型整体优劣；为避免把单一案例误读成排行榜，卡片不展示模型名称。"
        if zh
        else "The examples below come from the same-case comparison index generated from `artifacts/reports`. The input image, sample ID, and exact generation prompt are identical within each pair. Candidates were shortlisted by score quantiles and gap, then reviewed frame by frame. Labels describe only this metric—not overall model quality—and model names are intentionally omitted so a single case is not mistaken for a leaderboard."
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


def main() -> None:
    CURATED_ROOT.mkdir(parents=True, exist_ok=True)
    video_map, input_map = old_assets()
    selected = select_rows()
    for row in selected:
        row["input_asset"] = publish_input(row, input_map)
        row["high_asset"] = publish_video(row, "high", video_map)
        row["low_asset"] = publish_video(row, "low", video_map)

    audit = {
        "version": 1,
        "reviewed_at": "2026-08-01",
        "source_reports_root": "artifacts/reports",
        "sources": [
            {
                "path": "site/.cache/compare",
                "description": "same-case comparison cache derived from retained formal reports",
                "generated_at": "2026-07-17T22:53:22.381Z",
            },
            {
                "path": "artifacts/reports",
                "description": "fresh formal reports for background, segment continuity, and physical-FPS metrics",
            },
        ],
        "selection": "quantile shortlist followed by manual three-timepoint frame-strip review",
        "same_case_fields": ["suite", "sample_id", "source_path", "prompt"],
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

    print(
        json.dumps(
            {
                "published_examples": len(selected),
                "updated_doc_files": (len(by_file) + len(UNAVAILABLE_DOCS)) * 2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
