"""Model-independent, CPU-only eligibility preparation for JEDi references.

Never use this filter for predictions. A frozen eligible reference population
is strict at encode time: new decode failures still fail the evaluation.

Inspection is the same CPU probe as FVMD (ffprobe frame count, fps, sampled
windows). The roster is keyed by the JEDi clip spec, so window counts stay
JEDi's (up to 2 clips) and do not reuse the FVMD selection file.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from worldatlas.benchmark.distribution_clips import (
    ClipDecodeError,
    ClipSamplingSpec,
    atomic_write_text,
    clip_set_signature,
)
from worldatlas.benchmark import fvmd_reference as fvmd_ref
from worldatlas.benchmark.fvmd_reference import _digest, cache_lock, inspect_reference
from worldatlas.common.progress import log_progress


def prepare_reference_selection(entries, spec: ClipSamplingSpec, cache_dir: Path, *, workers=8):
    """Freeze a content-bound roster before loading any GPU model; reuse it.

    Only expected video eligibility errors are exclusions. Operational errors
    propagate and never publish a partial roster. The key includes ALL inputs,
    including excluded videos, but no prediction/model identity.
    """
    entries = sorted(entries, key=lambda item: (str(item[0]), item[1]))
    if len({str(p) for p, _ in entries}) != len(entries):
        raise ValueError("JEDi reference population contains duplicate paths")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("JEDi reference preparation requires ffprobe on PATH")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inspect_hash = hashlib.sha256(Path(fvmd_ref.__file__).read_bytes()).hexdigest()
    extra = ["jedi_reference_eligibility", source_hash, inspect_hash]
    signature = clip_set_signature(entries, spec, extra=extra)
    destination = cache_dir / "reference_selection" / f"{signature}.json"
    with cache_lock(destination):
        if destination.is_file():
            payload = json.loads(destination.read_text())
            report = payload["selection"]
            if payload["sha256"] != _digest(report) or report["input_signature"] != signature:
                raise ValueError(f"JEDi reference selection integrity mismatch: {destination}")
        else:
            version = subprocess.run([ffprobe, "-version"], capture_output=True, text=True,
                                     check=True, timeout=30).stdout.splitlines()[0]
            log_progress("jedi_reference_preflight", status="start", videos=len(entries), workers=workers)

            def inspect(entry):
                path, identifier = entry
                try:
                    details = inspect_reference(path, spec, ffprobe)
                except ClipDecodeError as exc:
                    details = dict(eligible=False, reason="decode_or_sampling_error", error=str(exc))
                return dict(path=str(path), id=identifier, **details)

            rows = []
            with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 32))) as pool:
                for row in pool.map(inspect, entries):
                    rows.append(row)
                    if len(rows) % 25 == 0 or len(rows) == len(entries):
                        log_progress("jedi_reference_preflight", videos=f"{len(rows)}/{len(entries)}")
            eligible = [row for row in rows if row["eligible"]]
            report = dict(schema_version=1, input_signature=signature, clip_spec=spec.describe(),
                          ffprobe_version=version, candidate_video_count=len(rows),
                          eligible_video_count=len(eligible), excluded_video_count=len(rows)-len(eligible),
                          clip_count=sum(row["clip_count"] for row in eligible), videos=rows)
            payload = dict(selection=report, sha256=_digest(report))
            if clip_set_signature(entries, spec, extra=extra) != signature:
                raise ValueError("Reference inputs changed during preparation; no roster published")
            atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not report["eligible_video_count"]:
        raise ValueError(f"No eligible JEDi references; inspect {destination}")
    log_progress("jedi_reference_ready", eligible=report["eligible_video_count"],
                 excluded=report["excluded_video_count"], signature=payload["sha256"], manifest=destination)
    selected = [(Path(row["path"]), row["id"]) for row in report["videos"] if row["eligible"]]
    verified = {row["path"]: row["decoded_frame_count"] for row in report["videos"] if row["eligible"]}
    return selected, verified, payload
