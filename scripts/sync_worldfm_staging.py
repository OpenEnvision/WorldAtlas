#!/usr/bin/env python3
"""Sync completed WorldFM intermediate artifacts from work_root to output_dir/_staging."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldarena.models.adapters.worldfm_decoupled_runner import (  # noqa: E402
    _publish_stage_staging,
    _stage_dir,
)


def _load_requests(work_root: Path) -> dict[str, dict[str, Any]]:
    requests_path = work_root / "requests.jsonl"
    if not requests_path.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    with requests_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            stem = str(request["prediction_stem"])
            mapping[stem] = request
    return mapping


def _request_for_stem(
    stem: str,
    *,
    output_dir: Path,
    requests_by_stem: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if stem in requests_by_stem:
        return requests_by_stem[stem]
    return {
        "prediction_stem": stem,
        "output_path": str(output_dir / f"{stem}.mp4"),
    }


def _iter_done_stems(work_root: Path, stage: str) -> list[str]:
    stage_root = work_root / stage
    if not stage_root.is_dir():
        return []
    stems: list[str] = []
    for sample_dir in stage_root.iterdir():
        if sample_dir.is_dir() and (sample_dir / "done.json").is_file():
            stems.append(sample_dir.name)
    return sorted(stems)


def _staging_done_stems(output_dir: Path, stage: str) -> set[str]:
    staging_root = output_dir / "_staging"
    if not staging_root.is_dir():
        return set()
    done_stems: set[str] = set()
    for sample_dir in staging_root.iterdir():
        if sample_dir.is_dir() and (sample_dir / stage / "done.json").is_file():
            done_stems.add(sample_dir.name)
    return done_stems


def _stems_to_sync(
    work_root: Path,
    output_dir: Path,
    stage: str,
    *,
    incremental: bool,
) -> list[str]:
    work_stems = _iter_done_stems(work_root, stage)
    if not incremental:
        return work_stems
    synced = _staging_done_stems(output_dir, stage)
    return [stem for stem in work_stems if stem not in synced]


def _count_done(work_root: Path, stage: str) -> int:
    return len(_iter_done_stems(work_root, stage))


def _sync_stage(
    *,
    work_root: Path,
    output_dir: Path,
    stage: str,
    requests_by_stem: dict[str, dict[str, Any]],
    report_every: int,
    incremental: bool,
) -> tuple[int, int]:
    stems = _stems_to_sync(work_root, output_dir, stage, incremental=incremental)
    total = len(stems)
    files_copied = 0
    for index, stem in enumerate(stems, start=1):
        request = _request_for_stem(stem, output_dir=output_dir, requests_by_stem=requests_by_stem)
        stage_dir = _stage_dir(work_root, stage, request)
        before = _count_stage_files(output_dir, stem, stage)
        _publish_stage_staging(work_root, request, stage, stage_dir)
        after = _count_stage_files(output_dir, stem, stage)
        files_copied += max(after - before, after)
        if index % report_every == 0 or index == total:
            print(
                f"[sync_worldfm_staging] {stage}: {index}/{total} samples synced",
                flush=True,
            )
    return total, files_copied


def _count_stage_files(output_dir: Path, stem: str, stage: str) -> int:
    dest = output_dir / "_staging" / stem / stage
    if not dest.is_dir():
        return 0
    return sum(1 for path in dest.iterdir() if path.is_file())


def _sync_final_outputs(
    *,
    work_root: Path,
    output_dir: Path,
    requests_by_stem: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    stems = _iter_done_stems(work_root, "final")
    copied = 0
    for stem in stems:
        request = _request_for_stem(stem, output_dir=output_dir, requests_by_stem=requests_by_stem)
        output_path = Path(str(request["output_path"])).expanduser().resolve()
        if output_path.exists():
            continue
        stage_dir = _stage_dir(work_root, "final", request)
        for candidate in (
            stage_dir / f"{stem}.mp4",
            stage_dir / "output.mp4",
            work_root / "final" / stem / f"{stem}.mp4",
        ):
            if candidate.is_file():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, output_path)
                copied += 1
                break
    return len(stems), copied


def _count_staging_samples(output_dir: Path) -> dict[str, int]:
    staging_root = output_dir / "_staging"
    if not staging_root.is_dir():
        return {"prepare": 0, "merge": 0, "any": 0}
    prepare = merge = any_stage = 0
    for sample_dir in staging_root.iterdir():
        if not sample_dir.is_dir():
            continue
        has_prepare = (sample_dir / "prepare" / "done.json").is_file()
        has_merge = (sample_dir / "merge" / "done.json").is_file()
        if has_prepare:
            prepare += 1
        if has_merge:
            merge += 1
        if has_prepare or has_merge:
            any_stage += 1
    return {"prepare": prepare, "merge": merge, "any": any_stage}


def _count_mp4(output_dir: Path) -> int:
    return sum(1 for path in output_dir.glob("*.mp4") if path.is_file())


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _update_generation_manifest(
    output_dir: Path,
    *,
    work_root: Path,
    migration_manifest: dict[str, Any],
) -> None:
    path = output_dir / "generation_manifest.json"
    manifest = _load_json(path) or {}
    manifest.update(
        {
            "model_name": manifest.get("model_name", "worldfm"),
            "family": manifest.get("family", "worldfm"),
            "output_dir": str(output_dir),
            "work_root": str(work_root),
            "total_samples": migration_manifest["total_samples"],
            "prepare_done": migration_manifest["prepare_done"],
            "merge_done": migration_manifest["merge_done"],
            "final_done": migration_manifest["final_done"],
            "staging_synced": migration_manifest["staging_synced"],
            "mp4_count": migration_manifest["mp4_count"],
            "migration_manifest": str(output_dir / "migration_manifest.json"),
            "staging_sync_time": migration_manifest["sync_time"],
        }
    )
    _write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync WorldFM staging artifacts to output_dir/_staging.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--total-samples", type=int, default=5000)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only sync samples missing done.json under output_dir/_staging.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    if not work_root.is_dir():
        raise SystemExit(f"work_root not found: {work_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[sync_worldfm_staging] output_dir={output_dir}", flush=True)
    print(f"[sync_worldfm_staging] work_root={work_root}", flush=True)

    requests_by_stem = _load_requests(work_root)
    prepare_done = _count_done(work_root, "prepare")
    merge_done = _count_done(work_root, "merge")
    final_done = _count_done(work_root, "final")

    print(
        f"[sync_worldfm_staging] work done counts: prepare={prepare_done} merge={merge_done} final={final_done}",
        flush=True,
    )

    prepare_synced, prepare_files = _sync_stage(
        work_root=work_root,
        output_dir=output_dir,
        stage="prepare",
        requests_by_stem=requests_by_stem,
        report_every=args.report_every,
        incremental=args.incremental,
    )
    merge_synced, merge_files = _sync_stage(
        work_root=work_root,
        output_dir=output_dir,
        stage="merge",
        requests_by_stem=requests_by_stem,
        report_every=args.report_every,
        incremental=args.incremental,
    )
    final_done_synced, final_copied = _sync_final_outputs(
        work_root=work_root,
        output_dir=output_dir,
        requests_by_stem=requests_by_stem,
    )

    staging_counts = _count_staging_samples(output_dir)
    mp4_count = _count_mp4(output_dir)
    elapsed = time.time() - started

    migration_manifest = {
        "total_samples": int(args.total_samples),
        "prepare_done": prepare_done,
        "merge_done": merge_done,
        "final_done": final_done,
        "staging_synced": staging_counts,
        "prepare_synced": prepare_synced,
        "merge_synced": merge_synced,
        "final_outputs_copied": final_copied,
        "mp4_count": mp4_count,
        "work_root": str(work_root),
        "output_dir": str(output_dir),
        "sync_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "elapsed_seconds": round(elapsed, 2),
        "files_touched_estimate": prepare_files + merge_files + final_copied,
    }
    manifest_path = output_dir / "migration_manifest.json"
    _write_json(manifest_path, migration_manifest)
    _update_generation_manifest(output_dir, work_root=work_root, migration_manifest=migration_manifest)

    print("[sync_worldfm_staging] summary:", flush=True)
    print(f"  prepare synced: {prepare_synced} (work done={prepare_done})", flush=True)
    print(f"  merge synced:   {merge_synced} (work done={merge_done})", flush=True)
    print(f"  staging dirs:   prepare={staging_counts['prepare']} merge={staging_counts['merge']}", flush=True)
    print(f"  final done:     {final_done_synced}, mp4 copied={final_copied}, mp4 in OUT={mp4_count}", flush=True)
    print(f"  manifest:       {manifest_path}", flush=True)
    print(f"  elapsed:        {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
