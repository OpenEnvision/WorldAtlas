"""Shared generation-record lookup for action and memory diagnostics."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from worldatlas.benchmark.schemas import BenchmarkSample


@lru_cache(maxsize=128)
def _load_generation_record_file(path_text: str, content_stamp: tuple | None = None) -> tuple[dict[str, Any], ...]:
    """Parse JSONL generation records from a cached file path."""
    path = Path(path_text)
    records: list[dict[str, Any]] = []
    if not path.exists():
        return ()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return tuple(records)


def _record_files_in_directory(directory: str, mtime_ns: int) -> tuple[Path, ...]:
    """Discover sidecars afresh; directory timestamps are not content evidence."""
    del mtime_ns  # Additions may share a timestamp on network filesystems.
    return tuple(sorted(Path(directory).glob("generation_records*.jsonl"),
                        key=lambda path: (".archive." in path.name, path.name)))


def _candidate_record_files(prediction_path: Path) -> list[Path]:
    """Search parent directories for generation_records JSONL sidecars."""
    directories: list[Path] = []
    current = prediction_path.resolve().parent
    for _ in range(3):
        directories.append(current)
        if current.parent == current:
            break
        current = current.parent

    files: list[Path] = []
    for directory in dict.fromkeys(directories):
        files.extend(_record_files_in_directory(str(directory), directory.stat().st_mtime_ns))
    return files


def _resolve_record_path(value: Any, *, base_dir: Path) -> Path | None:
    """Resolve a possibly relative generation-record path against a base dir."""
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        return path.resolve()
    except OSError:
        return path


@lru_cache(maxsize=128)
def _generation_record_index(path_text: str, content_stamp: tuple | None = None) -> tuple[dict, dict]:
    """Index the cached record snapshot once, avoiding repeated filesystem resolves.

    Path matches use the first record; sample fallbacks use the last eligible
    record, matching the original scan order. Like _load_generation_record_file,
    this cache lives for one evaluation process over completed predictions.
    """
    base_dir = Path(path_text).parent
    by_path: dict[Path, dict[str, Any]] = {}
    by_sample: dict[tuple[str, Any], tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(_load_generation_record_file(path_text, content_stamp)):
        if record.get("status") in {"skipped_existing", "failed", "pending"}:
            continue
        for key in ("prediction_path", "output_path"):
            record_path = _resolve_record_path(record.get(key), base_dir=base_dir)
            if record_path is not None:
                by_path.setdefault(record_path, record)
        sample_id = str(record.get("sample_id") or record.get("case_id") or "")
        by_sample[(sample_id, record.get("suite"))] = (index, record)
    return by_path, by_sample


def _find_generation_record(sample: BenchmarkSample, prediction_path: Path) -> dict[str, Any] | None:
    """Match a prediction video to its generation timing/action record."""
    if getattr(sample, "sample_id_policy", None):
        # A legacy same-name/sample-ID record is not provenance for migrated
        # pixels. Only use a record explicitly pinned by the verified binding.
        from worldatlas.benchmark.integrity import digest_json
        binding = sample.prediction_binding or {}
        record = binding.get("generation_record")
        if not isinstance(record, dict):
            return None
        if digest_json(record) != binding.get("generation_record_sha256"):
            raise ValueError("Verified generation record changed")
        record = dict(record)
        if not binding.get("predictions_root") or not binding.get("path"):
            return record
        # Action sidecars travel with the prediction directory.
        from worldatlas.benchmark.sample_identity import prediction_relative_path
        relative_video = prediction_relative_path(binding)
        root = prediction_path.resolve().parents[len(relative_video.parts) - 1]
        old_root = Path(binding["predictions_root"])
        for key in ("prediction_path", "output_path", "action_spec_path"):
            if record.get(key):
                path = Path(record[key])
                if path.is_absolute() and path.is_relative_to(old_root):
                    record[key] = str(root / path.relative_to(old_root))
        return record
    resolved_prediction = prediction_path.resolve()
    fallback_by_sample: dict[str, Any] | None = None
    for records_file in _candidate_record_files(prediction_path):
        # Keep the lexical parent for relative paths, including symlinked JSONL.
        from worldatlas.benchmark.sample_identity import file_sha256
        by_path, by_sample = _generation_record_index(str(records_file.absolute()),
                                                      (file_sha256(records_file),))
        if resolved_prediction in by_path:
            return dict(by_path[resolved_prediction])
        candidates = [by_sample[key] for key in ((sample.sample_id, None), (sample.sample_id, sample.suite)) if key in by_sample]
        if candidates:
            fallback_by_sample = dict(max(candidates, key=lambda item: item[0])[1])
    return fallback_by_sample


def _load_action_payload(record: dict[str, Any] | None, prediction_path: Path) -> dict[str, Any] | None:
    """Load inline or sidecar action JSON referenced by a generation record."""
    if not record:
        return None
    inline = record.get("action_payload")
    if isinstance(inline, dict):
        return dict(inline)
    action_spec_path = record.get("action_spec_path")
    if action_spec_path:
        resolved = _resolve_record_path(action_spec_path, base_dir=prediction_path.resolve().parent)
        if resolved is not None and resolved.exists():
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    return None
