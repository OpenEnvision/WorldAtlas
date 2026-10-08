"""Published leaderboard scores: normalize observations, then aggregate.

This module intentionally depends only on the standard library. Publication
must not import GPU runtimes or infer a raw mean by inverting a scored mean.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

REVISION = "per_sample_scored_mean_v1"
ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/leaderboard_scoring_v1.json"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clip(value):
    if not finite(value):
        raise ValueError(f"Non-finite measurement: {value!r}")
    return max(0.0, min(1.0, value))


def linear(raw, lower=0.0, upper=1.0, higher=True):
    if not finite(raw) or not upper > lower:
        raise ValueError("Invalid raw measurement or calibration bounds")
    value = (raw - lower) / (upper - lower)
    return clip(value if higher else 1 - value)


def score_observation(metric, output, policy):
    """Use raw measurements (including declared components), never a mean."""
    spec = policy["metrics"][metric]
    raw = output.get("raw")
    details = output.get("details") or {}
    kind = spec["kind"]
    if kind == "camera":
        # Native intended-camera raw is a dictionary of trajectory diagnostics;
        # its actual per-video direction score is recorded separately.
        if details.get("native_camera_target_policy") == "native_intended" or "native_intent_accuracy" in details:
            value = details.get("native_intent_accuracy")
            if not finite(value):
                raise ValueError("Native camera report lacks its direction-confidence measurement")
            return clip(value)
        if not isinstance(raw, dict):
            raise ValueError("Camera report lacks declared raw components")
        components = [raw[name] for name in spec["direct_pose_components"]]
        return math.sqrt(linear(components[0], 0, spec["rotation_upper"], False)
                         * linear(components[1], 0, spec["translation_upper"], False))
    if kind == "components":
        if not isinstance(raw, dict):
            raise ValueError(f"{metric}: expected raw component dictionary")
        values = [linear(raw[name], lo, hi, higher)
                  for name, lo, hi, higher in spec["components"]]
        return math.fsum(values) / len(values)
    if kind == "zscore":
        if not finite(raw):
            raise ValueError(f"{metric}: non-finite raw measurement")
        x = ((raw - spec["avg"]) / spec["std"] - spec["z_min"]) / (spec["z_max"] - spec["z_min"])
        return clip(spec["range"][0] + (spec["range"][1] - spec["range"][0]) * x)
    if kind == "soft_tail":
        if not finite(raw) or raw < 0:
            raise ValueError(f"{metric}: invalid distance")
        x = (spec["upper"] - raw) / (spec["upper"] - spec["lower"])
        tail = spec["soft_tail"]
        return clip(x if x >= tail else tail * (tail / (2 * tail - x)))
    if kind == "linear":
        return linear(raw, spec.get("lower", 0), spec.get("upper", 1), spec.get("higher", True))
    raise ValueError(f"Unknown calibration kind: {kind}")


def raw_mean(values):
    """Preserve native scalar/component units; do not manufacture a scalar."""
    if values and all(finite(v) for v in values):
        return math.fsum(values) / len(values)
    if values and all(isinstance(v, dict) for v in values):
        keys = set(values[0])
        if all(set(v) == keys and all(finite(x) for x in v.values()) for v in values):
            return {k: math.fsum(v[k] for v in values) / len(values) for k in sorted(keys)}
    return None


def resolve_report(root, source):
    value = source.get("reportDir") or source.get("report")
    if not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.name in {"per_metric_summary.json", "diagnostic_metric_summary.json"}:
        path = path.parent
    return path


def aggregate_report(path, suite, metrics, policy):
    """Stream once per report and keep all numeric statuses, including old diagnostic."""
    path = Path(path)
    metrics = set(metrics)
    result = {}
    distribution = metrics & {"jedi", "fvmd"}
    if distribution:
        file = path / "distribution_metric_details.json"
        if file.is_file():
            data = read(file).get(suite, {})
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            for metric in distribution:
                value = data.get(metric) or {}
                if finite(value.get("raw")):
                    result[metric] = {"raw_mean": value["raw"],
                        "scored_mean": score_observation(metric, value, policy),
                        "aggregation": "suite_distribution_once", "numeric_count": 1,
                        "source_file": str(file), "source_sha256": sha,
                        "report_normalized_mean": value.get("normalized")}
    metrics -= distribution
    if not metrics:
        return result
    if path.is_file() and path.suffix == ".json":
        task = read(path)
        if metrics != {"memory_scene_f1"} or "samples" not in task:
            raise ValueError(f"Unsupported task result: {path}")
        rows = task["samples"]
        values = [r["task_score"] for r in rows if finite(r.get("task_score"))]
        if values:
            mean = math.fsum(values) / len(values)
            result["memory_scene_f1"] = {"raw_mean": mean, "scored_mean": math.fsum(
                score_observation("memory_scene_f1", {"raw": v}, policy) for v in values) / len(values),
                "numeric_count": len(values), "expected_count": len(rows),
                "aggregation": "fixed_gate_task_mean", "report_normalized_mean": mean,
                "source_file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return result
    file = path if path.name.endswith(".jsonl") else path / "per_sample_scores.jsonl"
    if not file.is_file():
        return result
    values = defaultdict(list)
    raws = defaultdict(list)
    normalized = defaultdict(list)
    strata = defaultdict(lambda: defaultdict(list))
    ids = defaultdict(set)
    excluded = defaultdict(int)
    count = 0
    sha = hashlib.sha256()
    with file.open("rb") as stream:
        for line in stream:
            sha.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != suite:
                continue
            count += 1
            for metric in metrics:
                output = (row.get("metrics") or {}).get(metric) or {}
                if metric == "physical_fps_error" and not output:
                    # Older Pulse reports keep the absolute-error components in
                    # the percentage/CV payload instead of duplicating the metric.
                    pulse = next(((row.get("metrics") or {}).get(k) for k in
                                  ("physical_fps_pct_error", "physical_fps_intra_video_cv")
                                  if (row.get("metrics") or {}).get(k)), {})
                    d = pulse.get("details") or {}
                    if finite(d.get("avg_phyfps")) and finite(d.get("meta_fps")):
                        output = {"raw": abs(d["avg_phyfps"] - d["meta_fps"]), "details": d}
                observed_gate_failure = (metric == "memory_return_gate"
                    and output.get("raw") == 0 and output.get("normalized") == 0
                    and (output.get("details") or {}).get("measurement_status") == "observed_pose_track")
                if not output or (output.get("error") and not observed_gate_failure) or output.get("raw") is None:
                    excluded[metric] += 1
                    continue
                sid = row.get("sample_id")
                if not sid or sid in ids[metric]:
                    raise ValueError(f"Missing/duplicate sample ID: {file}:{metric}:{sid}")
                ids[metric].add(sid)
                value = score_observation(metric, output, policy)
                values[metric].append(value)
                raws[metric].append(output["raw"])
                if finite(output.get("normalized")):
                    normalized[metric].append(output["normalized"])
                key = json.dumps([row.get("style"), row.get("environment"), row.get("scene"),
                                  row.get("motion_category")], ensure_ascii=False)
                strata[metric][key].append(value)
    for metric, scores in values.items():
        result[metric] = {"raw_mean": raw_mean(raws[metric]),
            "scored_mean": math.fsum(scores) / len(scores),
            "aggregation": "per_sample_then_mean", "numeric_count": len(scores),
            "expected_count": count, "excluded_count": excluded[metric],
            "sample_ids_sha256": digest(sorted(ids[metric])),
            "report_normalized_mean": (math.fsum(normalized[metric]) / len(normalized[metric])
                                       if len(normalized[metric]) == len(scores) else None),
            "strata": {k: {"count": len(v), "scored_mean": math.fsum(v) / len(v)}
                       for k, v in sorted(strata[metric].items())},
            "source_file": str(file), "source_sha256": sha.hexdigest()}
    return result


def prepare_publication(payload, root=ROOT):
    """Recompute changed report cells; reuse explicitly bound canonical cells.

    Legacy migration requires the repair command and its provenance manifest.
    A writer cannot silently publish error_ref summary numbers into schema v2.
    """
    root = Path(root)
    if payload.get("schema_version") != 2:
        raise ValueError("Leaderboard needs explicit per-sample score migration before publication")
    policy = read(root / "config/leaderboard_scoring_v1.json")
    policy_sha = digest(policy)
    if payload.get("score_calibration", {}).get("revision") != REVISION:
        raise ValueError("Unsupported leaderboard calibration; refusing mixed score scales")
    if payload["score_calibration"].get("policy_sha256") != policy_sha:
        raise ValueError("Frozen calibration changed; an explicit score migration is required")
    verified_sources = {}
    pending = defaultdict(list)
    for suite, block in payload["suites"].items():
        for key, row in block["models"].items():
            details = row.setdefault("score_details", {})
            for metric in list(details):
                if not finite(row.get("scores", {}).get(metric)):
                    details.pop(metric)
            for metric, value in row.get("scores", {}).items():
                if value is None:
                    continue
                if not finite(value):
                    raise ValueError(f"Invalid published score: {suite}/{key}/{metric}")
                source = row.get("metric_sources", {}).get(metric) or {}
                old = details.get(metric) or {}
                binding = digest(source)
                if (old.get("revision") == REVISION and old.get("policy_sha256") == policy_sha
                        and old.get("source_binding") == binding
                        and value in (old.get("scored_mean"), old.get("input_score"))):
                    if not finite(old.get("scored_mean")) or not 0 <= old["scored_mean"] <= 1:
                        raise ValueError(f"Invalid cached score: {suite}/{key}/{metric}")
                    file = Path(old["source_file"])
                    file = file if file.is_absolute() else root / file
                    if file not in verified_sources:
                        with file.open("rb") as stream:
                            sha = hashlib.sha256()
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                sha.update(chunk)
                            verified_sources[file] = sha.hexdigest()
                    if verified_sources[file] != old.get("source_sha256"):
                        raise ValueError(f"Published source changed; publish an explicitly updated report source: {file}")
                    row["scores"][metric] = old["scored_mean"]
                    continue
                report = resolve_report(root, source)
                if report is None and suite in {"long_sequence", "long_sequence_30s"}:
                    report = resolve_report(root, row.get("memory_protocol") or {})
                if report is None:
                    raise ValueError(f"Unprovenanced score update: {suite}/{key}/{metric}")
                pending[(report, "memory_loop" if suite in {"long_sequence", "long_sequence_30s"} else suite)].append(
                    (row, metric, value, binding))
    for (report, suite), cells in pending.items():
        stats = aggregate_report(report, suite, [m for _, m, _, _ in cells], policy)
        for row, metric, value, binding in cells:
            if metric not in stats:
                raise ValueError(f"Missing raw measurements: {report}/{suite}/{metric}")
            stat = stats[metric]
            row["scores"][metric] = stat["scored_mean"]
            row["score_details"][metric] = {**stat, "revision": REVISION,
                "policy_sha256": policy_sha, "source_binding": binding, "input_score": value}
    return payload
