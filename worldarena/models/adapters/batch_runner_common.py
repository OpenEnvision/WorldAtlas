from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def add_common_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--generation_config_path", required=True, type=str)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_batch_spec(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty batch spec: {path}")
    return rows


def copy_output(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing generated output: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        temporary = destination.with_name(destination.name + ".tmp.syncing")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def latest_mp4(directory: Path) -> Path:
    candidates = sorted(
        [path for path in directory.rglob("*.mp4") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no mp4 outputs under {directory}")
    return candidates[0]


def print_status(sample_id: str, status: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"sample_id": sample_id, "status": status, **payload},
            ensure_ascii=False,
        ),
        flush=True,
    )
