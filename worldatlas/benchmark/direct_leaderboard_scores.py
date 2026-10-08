"""Four report-derived display scores, without changing legacy report storage.

`scores` retains publisher inputs. `score_details[m].published_score` is the
only display/aggregation value for these four metrics after migration. Historical
linear summaries remain explicitly historical: never label them raw/scored means.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

# Frozen evaluation environments must load the live, standard-library scorer.
_spec = importlib.util.spec_from_file_location("_direct_score_core", Path(__file__).with_name("leaderboard_scores.py"))
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
finite, raw_mean, score_observation = _core.finite, _core.raw_mean, _core.score_observation

ROOT = Path(__file__).resolve().parents[2]
REVISION = "direct_metric_mean_v1"
POLICY_FILE = "config/leaderboard_direct_scoring_v1.json"
METRICS = ("camera_error", "met3r_consistency", "optical_flow_aepe", "style_consistency")
LEGACY_SCALES = {"camera_error": (0.3, 1.0), "met3r_consistency": (0.3, 2.0),
                 "optical_flow_aepe": (0.2, 1.192), "style_consistency": (0.0004, 0.007)}


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_state(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": str(stat.st_mtime_ns)}


def source_file(root, source):
    value = source.get("report") or source.get("reportDir")
    if not value:
        return None
    p = Path(value)
    p = p if p.is_absolute() else Path(root) / p
    if p.suffix == ".jsonl":
        return p
    return (p.parent if p.suffix == ".json" else p) / "per_sample_scores.jsonl"


def forward_legacy(metric, linear_mean):
    """Replay a documented old display transform; never invert a mean."""
    ref, maximum = LEGACY_SCALES[metric]
    return 1 / (1 + maximum * (1 - linear_mean) / ref)


def matches(left, right):
    return finite(left) and finite(right) and abs(left - right) <= 0.000051


def input_matches(metric, stored, report_mean):
    return finite(report_mean) and (matches(stored, report_mean)
        or matches(stored, forward_legacy(metric, report_mean))
        or matches(stored, forward_legacy(metric, round(report_mean, 4))))


def aggregate(path, suite, metrics, policy):
    """Score valid measurements, preserving zero and excluding failed/NA rows."""
    path = Path(path)
    values, raws, normalized, error_ref = (defaultdict(list) for _ in range(4))
    ids = defaultdict(set)
    kinds = defaultdict(set)
    expected = 0
    h = hashlib.sha256()
    before = file_state(path)
    with path.open("rb") as stream:
        for line in stream:
            h.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != suite:
                continue
            expected += 1
            for metric in metrics:
                output = (row.get("metrics") or {}).get(metric) or {}
                if (output.get("error") or output.get("eligibility_status") in
                        {"not_applicable", "failed", "missing"} or not finite(output.get("normalized"))):
                    continue
                sid = row.get("sample_id")
                if not sid or sid in ids[metric]:
                    raise ValueError(f"Missing/duplicate sample: {path}/{metric}/{sid}")
                value = score_observation(metric, output, policy)
                values[metric].append(value)
                raws[metric].append(output.get("raw"))
                normalized[metric].append(output["normalized"])
                ref, maximum = LEGACY_SCALES[metric]
                error = 1 - value if metric == "camera_error" else max(0, output["raw"])
                error_ref[metric].append(1 / (1 + error / ref))
                ids[metric].add(sid)
                if metric == "camera_error":
                    details = output.get("details") or {}
                    kinds[metric].add("native_direction_confidence" if "native_intent_accuracy" in details
                                      else "direct_pose_components")
    if file_state(path) != before:
        raise ValueError(f"Report changed while reading: {path}")
    result = {}
    for metric, scores in values.items():
        if len(kinds[metric]) > 1:
            raise ValueError(f"Mixed camera definitions in {path}")
        result[metric] = {
            "published_score": math.fsum(scores) / len(scores),
            "scored_mean": math.fsum(scores) / len(scores),
            "raw_mean": raw_mean(raws[metric]),
            "measurement": next(iter(kinds[metric]), "raw_distance"),
            "aggregation": "per_sample_then_mean", "numeric_count": len(scores),
            "expected_count": expected, "excluded_count": expected - len(scores),
            "report_normalized_mean": math.fsum(normalized[metric]) / len(scores),
            "report_error_ref_mean": math.fsum(error_ref[metric]) / len(scores),
            "sample_ids_sha256": hashlib.sha256(json.dumps(sorted(ids[metric])).encode()).hexdigest(),
            "source_file": str(path.resolve()), "source_sha256": h.hexdigest(), "source_state": before,
        }
    return result


def calibration(root=ROOT):
    return {"revision": REVISION, "policy": POLICY_FILE,
            "policy_sha256": sha256(Path(root) / POLICY_FILE), "metrics": list(METRICS),
            "score_field": "score_details[metric].published_score",
            "legacy_policy": "Explicitly retain archived published linear summaries; raw/scored means unavailable"}


def bind(stat, row, metric, policy_sha):
    return {**stat, "revision": REVISION, "policy_sha256": policy_sha,
            "input_score": row["scores"][metric],
            "source_snapshot": deepcopy(row.get("metric_sources", {}).get(metric) or {})}


def validate_detail(detail, row, metric, policy_sha):
    if (detail.get("revision") != REVISION or detail.get("policy_sha256") != policy_sha
            or detail.get("input_score") != row["scores"][metric]
            or detail.get("source_snapshot") != (row.get("metric_sources", {}).get(metric) or {})
            or not finite(detail.get("published_score")) or not 0 <= detail["published_score"] <= 1
            or len(detail.get("source_sha256", "")) != 64):
        raise ValueError(f"Stale or invalid direct score: {metric}")
    if detail.get("aggregation") == "per_sample_then_mean":
        if detail.get("scored_mean") != detail["published_score"] or not detail.get("numeric_count"):
            raise ValueError(f"Invalid scored mean: {metric}")
    elif detail.get("aggregation") == "retained_published_linear_summary":
        if detail.get("raw_mean") is not None or detail.get("scored_mean") is not None:
            raise ValueError(f"Historical summary cannot claim a raw/scored mean: {metric}")
    else:
        raise ValueError(f"Unknown score aggregation: {metric}")


def prepare_publication(payload, root=ROOT):
    """Refresh changed cells from their explicit report; retain valid bound cells.

    Old, un-migrated fixtures and reports are left alone. The server rejects a
    missing overlay once direct_scoring is enabled, preventing silent scale drift.
    """
    root = Path(root)
    spec = payload.get("score_calibration", {}).get("direct_scoring")
    if spec is None:
        return payload
    if spec != calibration(root):
        raise ValueError("Direct scoring policy changed; explicit migration required")
    policy = read(root / POLICY_FILE)
    pending = defaultdict(list)
    checked = {}
    for suite, block in payload.get("suites", {}).items():
        for model, row in block.get("models", {}).items():
            for metric in METRICS:
                value = row.get("scores", {}).get(metric)
                if value is None:
                    row.get("score_details", {}).pop(metric, None)
                    continue
                if not finite(value):
                    raise ValueError(f"Invalid input score: {suite}/{model}/{metric}")
                detail = row.get("score_details", {}).get(metric, {})
                source = row.get("metric_sources", {}).get(metric) or {}
                if (detail.get("input_score") == value and detail.get("source_snapshot") == source
                        and detail.get("revision") == REVISION):
                    validate_detail(detail, row, metric, spec["policy_sha256"])
                    f = Path(detail["source_file"])
                    f = f if f.is_absolute() else root / f
                    if f not in checked:
                        checked[f] = (file_state(f), sha256(f))
                    if checked[f][1] != detail["source_sha256"]:
                        raise ValueError(f"Published evidence changed: {f}; publish an updated report source")
                    continue
                f = source_file(root, source)
                if f is None or not f.is_file():
                    raise ValueError(f"New direct score requires per-sample evidence: {suite}/{model}/{metric}")
                pending[(f, suite)].append((row, metric))
    for (f, suite), cells in pending.items():
        stats = aggregate(f, suite, {m for _, m in cells}, policy)
        for row, metric in cells:
            stat = stats.get(metric)
            if not stat or not (input_matches(metric, row["scores"][metric], stat["report_normalized_mean"])
                               or matches(row["scores"][metric], stat["report_error_ref_mean"])):
                raise ValueError(f"Published input does not match report: {f}/{metric}")
            row.setdefault("score_details", {})[metric] = bind(stat, row, metric, spec["policy_sha256"])
    return payload
