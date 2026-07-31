#!/usr/bin/env python3
"""Render compact frame strips for human review of metric example pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


FRAME_W = 256
FRAME_H = 144
LABEL_W = 150
GAP = 8
BG = (241, 245, 249)
INK = (15, 23, 42)
MUTED = (71, 85, 105)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = f"{current} {word}".strip()
        if len(proposal) <= width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def video_frames(path: Path) -> list[Image.Image]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    frame_count = int(stream.frames or 0)
    if frame_count <= 0:
        decoded = list(container.decode(stream))
        frame_count = len(decoded)
    else:
        decoded = container.decode(stream)
    targets = {round((frame_count - 1) * ratio) for ratio in (0.12, 0.5, 0.88)}
    frames: list[Image.Image] = []
    for index, frame in enumerate(decoded):
        if index not in targets:
            continue
        image = frame.to_image().convert("RGB")
        image.thumbnail((FRAME_W, FRAME_H), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (FRAME_W, FRAME_H), (15, 23, 42))
        canvas.paste(image, ((FRAME_W - image.width) // 2, (FRAME_H - image.height) // 2))
        frames.append(canvas)
    container.close()
    if len(frames) != 3:
        raise RuntimeError(f"expected 3 review frames, decoded {len(frames)} from {path}")
    return frames


def remap_cpfs(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    prefix = "/mnt/cpfs/"
    if path.startswith(prefix):
        remapped = Path("/mnt/cpfsB") / path.removeprefix(prefix)
        if remapped.is_file():
            return remapped
    return candidate


def render(row: dict, output: Path) -> None:
    source = Image.open(remap_cpfs(row["source_path"])).convert("RGB")
    source.thumbnail((FRAME_W, FRAME_H), Image.Resampling.LANCZOS)
    input_frame = Image.new("RGB", (FRAME_W, FRAME_H), (15, 23, 42))
    input_frame.paste(source, ((FRAME_W - source.width) // 2, (FRAME_H - source.height) // 2))
    high_frames = video_frames(remap_cpfs(row["high"]["prediction_path"]))
    low_frames = video_frames(remap_cpfs(row["low"]["prediction_path"]))

    width = LABEL_W + GAP + FRAME_W * 3 + GAP * 2 + 24
    header_h = 132
    height = header_h + FRAME_H * 3 + GAP * 2 + 24
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    title_font = font(22, bold=True)
    body_font = font(14)
    label_font = font(16, bold=True)

    draw.text((12, 10), f"{row['metric']}  |  {row['sample_id']}", fill=INK, font=title_font)
    prompt_lines = wrap(row["prompt"], 112)[:3]
    for index, line in enumerate(prompt_lines):
        draw.text((12, 42 + index * 18), line, fill=MUTED, font=body_font)
    draw.text(
        (12, 102),
        f"high {row['high']['normalized']:.4f}  /  low {row['low']['normalized']:.4f}  /  gap {row['gap']:.4f}",
        fill=INK,
        font=body_font,
    )

    rows = (("INPUT", [input_frame] * 3), ("HIGH", high_frames), ("LOW", low_frames))
    for row_index, (label, images) in enumerate(rows):
        y = header_h + row_index * (FRAME_H + GAP)
        color = (22, 101, 52) if label == "HIGH" else (153, 27, 27) if label == "LOW" else INK
        draw.text((12, y + FRAME_H // 2 - 10), label, fill=color, font=label_font)
        for image_index, frame in enumerate(images):
            x = LABEL_W + image_index * (FRAME_W + GAP)
            sheet.paste(frame, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shortlist", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-metric", type=int, default=3)
    args = parser.parse_args()

    rows = json.loads(args.shortlist.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    results = []
    for row in rows:
        key = (row["suite"], row["metric"])
        if counts[key] >= args.per_metric:
            continue
        rank = counts[key] + 1
        output = args.output_dir / f"{row['metric']}__{rank:02d}__{row['sample_id']}.jpg"
        try:
            render(row, output)
        except Exception as error:  # keep reviewing other candidates
            results.append({"status": "error", "output": str(output), "error": str(error)})
            continue
        counts[key] += 1
        results.append({"status": "ok", "output": str(output), "metric": row["metric"], "rank": rank})

    (args.output_dir / "render_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rendered": sum(r["status"] == "ok" for r in results), "errors": sum(r["status"] == "error" for r in results)}, indent=2))


if __name__ == "__main__":
    main()
