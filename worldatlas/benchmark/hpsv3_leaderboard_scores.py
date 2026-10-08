"""Publish HPSv3 by stretching the mean of saved official-percentile raw scores.

Per-sample normalized scores may already be clipped; their mean cannot recover
the raw mean. Published values live in scores, with evidence in score_details.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRIC = "hpsv3_norm"
REVISION = "hpsv3_raw_mean_then_display_v1"
POLICY_FILE = "config/leaderboard_hpsv3_scoring_v1.json"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(value, decimals=4):
    scale = 10 ** decimals
    return math.floor(value * scale + 0.5) / scale


def score_mean(mean, policy, mean_rounding_decimals=None):
    if not finite(mean) or not 0 <= mean <= 1:
        raise ValueError("HPSv3 requires a mean on the official [0,1] raw scale")
    if mean_rounding_decimals is not None:
        if mean_rounding_decimals != 4:
            raise ValueError("Unsupported historical HPSv3 mean rounding")
        mean = rounded(mean, mean_rounding_decimals)
    value = (mean - policy["display_lower"]) / (policy["display_upper"] - policy["display_lower"])
    return rounded(max(0, min(1, value)), policy["storage_decimals"])


def calibration(root=ROOT):
    return {"revision": REVISION, "policy": POLICY_FILE,
            "policy_sha256": sha256(Path(root) / POLICY_FILE), "metric": METRIC,
            "score_field": "scores.hpsv3_norm", "aggregation": "mean_raw_then_display"}


def source_file(root, source):
    value = source.get("report") or source.get("reportDir")
    if not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else Path(root) / path
    if path.suffix == ".jsonl":
        return path
    return (path.parent if path.suffix == ".json" else path) / "per_sample_scores.jsonl"


def aggregate(path, suite, policy):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    raw, normalized, ids = [], [], set()
    expected = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != suite:
                continue
            expected += 1
            obs = (row.get("metrics") or {}).get(METRIC) or {}
            if (row.get("error") or obs.get("error") or obs.get("eligibility_status") in
                    {"failed", "missing", "not_applicable"} or not finite(obs.get("normalized"))):
                continue
            sid = row.get("sample_id")
            if not sid or sid in ids:
                raise ValueError(f"Missing/duplicate HPSv3 sample: {path}/{sid}")
            if not finite(obs.get("raw")) or not 0 <= obs["raw"] <= 1:
                raise ValueError(f"Missing/invalid HPSv3 raw observation: {path}/{sid}")
            if (obs.get("details") or {}).get("hpsv3_normalization_percentiles") != policy["official_percentiles"]:
                raise ValueError(f"HPSv3 raw normalization differs: {path}/{sid}")
            ids.add(sid)
            raw.append(obs["raw"])
            normalized.append(obs["normalized"])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"HPSv3 report changed while reading: {path}")
    if not raw:
        raise ValueError(f"No valid HPSv3 raw observations: {path}/{suite}")
    mean = math.fsum(raw) / len(raw)
    return {"aggregation": "mean_raw_then_display", "raw_mean": mean,
            "published_score": score_mean(mean, policy), "numeric_count": len(raw),
            "expected_count": expected, "excluded_count": expected - len(raw),
            "report_normalized_mean": math.fsum(normalized) / len(raw),
            "sample_ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            "source_file": str(path.resolve()), "source_sha256": digest.hexdigest(),
            "source_state": {"size": before.st_size, "mtime_ns": str(before.st_mtime_ns)}}


def bind(stat, row, policy_sha):
    return {**stat, "revision": REVISION, "policy_sha256": policy_sha,
            "source_snapshot": deepcopy(row.get("metric_sources", {}).get(METRIC) or {})}


def validate_detail(detail, row, policy, policy_sha):
    if (detail.get("revision") != REVISION or detail.get("policy_sha256") != policy_sha
            or detail.get("source_snapshot") != (row.get("metric_sources", {}).get(METRIC) or {})
            or not finite(detail.get("published_score"))
            or detail["published_score"] != row["scores"][METRIC]
            or len(detail.get("source_sha256", "")) != 64):
        raise ValueError("Stale or invalid HPSv3 publication")
    if detail.get("aggregation") == "mean_raw_then_display":
        if not detail.get("numeric_count"):
            raise ValueError("HPSv3 raw mean requires valid samples")
        mean = detail.get("raw_mean")
    elif detail.get("aggregation") == "retained_historical_mean_then_display":
        if detail.get("raw_mean") is not None or detail.get("numeric_count") is not None:
            raise ValueError("Historical HPSv3 summary cannot claim per-sample evidence")
        mean = detail.get("historical_mean")
    else:
        raise ValueError("Unsupported HPSv3 aggregation")
    if score_mean(mean, policy, detail.get("mean_rounding_decimals")) != detail["published_score"]:
        raise ValueError("HPSv3 published score does not match mean-first formula")


def prepare_publication(payload, root=ROOT):
    root = Path(root)
    spec = payload.get("score_calibration", {}).get("hpsv3_scoring")
    if spec is None:
        return payload
    if spec != calibration(root):
        raise ValueError("HPSv3 scoring policy changed; explicit migration required")
    policy = read(root / POLICY_FILE)
    checked = {}
    for suite, block in payload.get("suites", {}).items():
        for model, row in block.get("models", {}).items():
            value = row.get("scores", {}).get(METRIC)
            if value is None:
                row.get("score_details", {}).pop(METRIC, None)
                continue
            if not finite(value):
                raise ValueError(f"Invalid HPSv3 input: {suite}/{model}")
            detail = row.get("score_details", {}).get(METRIC) or {}
            source = row.get("metric_sources", {}).get(METRIC) or {}
            if detail.get("revision") == REVISION and detail.get("source_snapshot") == source:
                # The pinned source and scoring metadata are authoritative even
                # if a legacy importer has overwritten just the numeric value.
                bound = {**row, "scores": {**row["scores"], METRIC: detail.get("published_score")}}
                validate_detail(detail, bound, policy, spec["policy_sha256"])
                path = Path(detail["source_file"])
                path = path if path.is_absolute() else root / path
                if path not in checked:
                    checked[path] = sha256(path)
                if checked[path] != detail["source_sha256"]:
                    raise ValueError(f"Published HPSv3 evidence changed: {path}; update its source binding")
                row["scores"][METRIC] = detail["published_score"]
                continue
            path = source_file(root, source)
            if path is None or not path.is_file():
                raise ValueError(f"New HPSv3 publication requires per-sample raw evidence: {suite}/{model}")
            stat = aggregate(path, suite, policy)
            row["scores"][METRIC] = stat["published_score"]
            row.setdefault("score_details", {})[METRIC] = bind(stat, row, spec["policy_sha256"])
    return payload
