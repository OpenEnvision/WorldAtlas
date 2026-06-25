#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _media_probe(path: Path) -> dict[str, Any]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 0.0)
        frames = reader.count_frames()
    finally:
        reader.close()
    duration = frames / fps if fps else None
    return {
        "path": str(path),
        "frames": int(frames),
        "fps": fps,
        "duration_seconds": duration,
        "size": list(meta.get("size") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Warp-as-History WorldArena image_static run.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-duration", default=5.0, type=float)
    parser.add_argument("--duration-samples", default=16, type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    expected = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("suite") == "image_static":
                expected += 1

    mp4s = sorted(output_dir.glob("*.mp4"))
    batch_logs = sorted((output_dir / "_batch_specs").glob("*.log"))
    records = []
    per_log = []
    for log_path in batch_logs:
        generated = 0
        failed = 0
        last_sample_id = None
        for row in _json_lines(log_path):
            status = row.get("status")
            if status == "generated":
                generated += 1
                last_sample_id = row.get("sample_id")
            elif status == "failed":
                failed += 1
                last_sample_id = row.get("sample_id")
            records.append(row)
        per_log.append(
            {
                "log": str(log_path),
                "generated": generated,
                "failed": failed,
                "last_sample_id": last_sample_id,
            }
        )

    duration_checks = []
    if mp4s and args.duration_samples > 0:
        if len(mp4s) <= args.duration_samples:
            sample_paths = mp4s
        else:
            indexes = {
                round(i * (len(mp4s) - 1) / max(args.duration_samples - 1, 1))
                for i in range(args.duration_samples)
            }
            sample_paths = [mp4s[index] for index in sorted(indexes)]
        for path in sample_paths:
            try:
                duration_checks.append(_media_probe(path))
            except Exception as exc:
                duration_checks.append({"path": str(path), "error": str(exc)})

    generated = sum(1 for row in records if row.get("status") == "generated")
    failed = sum(1 for row in records if row.get("status") == "failed")
    short_videos = [
        item
        for item in duration_checks
        if item.get("duration_seconds") is not None and float(item["duration_seconds"]) < args.min_duration
    ]
    missing_duration = [item for item in duration_checks if item.get("error")]
    summary = {
        "output_dir": str(output_dir),
        "expected_image_static": expected,
        "mp4_count": len(mp4s),
        "generated_log_count": generated,
        "failed_log_count": failed,
        "batch_logs": len(batch_logs),
        "per_log": per_log,
        "duration_samples": duration_checks,
        "min_duration_seconds": args.min_duration,
        "short_duration_samples": short_videos,
        "duration_probe_errors": missing_duration,
        "complete": len(mp4s) == expected and generated == expected and failed == 0,
    }
    summary_path = output_dir / "verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary["complete"]:
        return 1
    if args.strict and (short_videos or missing_duration):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
