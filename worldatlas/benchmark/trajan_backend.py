"""Official TRAJAN per-video motion-quality backend."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.common.checkpoints import checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAPNET_REPO_CANDIDATES = (PROJECT_ROOT / "thirdparty" / "tapnet",)
AJ_THRESHOLDS = (1, 2, 4, 8, 16)
_RUNTIME_CACHE: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_tapnet_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = _resolve_project_path(configured)
        if not (repo_root / "tapnet").is_dir():
            raise FileNotFoundError(
                f"TRAJAN repo_root does not contain the tapnet package: {repo_root}"
            )
        return repo_root
    for candidate in DEFAULT_TAPNET_REPO_CANDIDATES:
        if (candidate / "tapnet").is_dir():
            return candidate.resolve()
    return None


def _resolve_checkpoint(runtime: dict[str, Any], key: str, filename: str) -> Path:
    env_key = {
        "tracker_checkpoint": "WORLDATLAS_TRAJAN_TRACKER_CHECKPOINT",
        "autoencoder_checkpoint": "WORLDATLAS_TRAJAN_AUTOENCODER_CHECKPOINT",
    }[key]
    configured = os.environ.get(env_key, "").strip() or runtime.get(key)
    path = (
        _resolve_project_path(configured)
        if configured
        else checkpoint_path("TRAJAN", filename, kind="file", required=False)
    )
    if not path.is_file():
        raise MetricLoadError(
            f"TRAJAN checkpoint missing: {path}. Run "
            "scripts/download_metric_checkpoints.sh --all."
        )
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        raise ValueError(f"TRAJAN autoencoder checkpoint is not an NPZ mapping: {path}")
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _recover_tree(flat_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for key, value in flat_dict.items():
        node = tree
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def _prepare_jax_environment(runtime: dict[str, Any]) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if os.environ.get("CUDA_ROOT"):
        return
    configured = runtime.get("cuda_root") or os.environ.get("CUDA_HOME")
    candidates = [Path(str(configured)).expanduser()] if configured else []
    candidates.append(Path("/usr/local/cuda"))
    for candidate in candidates:
        if candidate.is_dir():
            os.environ["CUDA_ROOT"] = str(candidate.resolve())
            return


def _prepare_python_compatibility() -> None:
    import typing

    if not hasattr(typing, "NotRequired"):
        from typing_extensions import NotRequired

        typing.NotRequired = NotRequired  # type: ignore[attr-defined]


def _load_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    repo_root = _resolve_tapnet_repo_root(runtime)
    if repo_root is not None and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tracker_checkpoint = _resolve_checkpoint(
        runtime,
        "tracker_checkpoint",
        "bootstapir_checkpoint_v2.npy",
    )
    autoencoder_checkpoint = _resolve_checkpoint(
        runtime,
        "autoencoder_checkpoint",
        "track_autoencoder_ckpt.npz",
    )
    tracker_chunk_size = _as_positive_int(runtime.get("tracker_query_chunk_size"), 32)
    decoder_chunk_size = _as_positive_int(runtime.get("decoder_chunk_size"), 32)
    model_type = str(runtime.get("tracker_model_type") or "bootstapir").lower()
    if model_type != "bootstapir":
        raise ValueError("TRAJAN official protocol requires tracker_model_type=bootstapir")

    cache_key = (
        str(repo_root),
        str(tracker_checkpoint),
        str(autoencoder_checkpoint),
        tracker_chunk_size,
        decoder_chunk_size,
    )
    cached = _RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _prepare_jax_environment(runtime)
    _prepare_python_compatibility()
    try:
        import jax
        from tapnet.models import tapir_model
        from tapnet.trajan import track_autoencoder
        from tapnet.utils import model_utils
    except Exception as exc:  # pragma: no cover - optional heavy runtime
        raise MetricLoadError(
            "TRAJAN runtime is unavailable. Install the official google-deepmind/tapnet "
            "package with JAX, Flax, Chex, and Einops. "
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        tracker_state = np.load(tracker_checkpoint, allow_pickle=True).item()
        tracker = tapir_model.ParameterizedTAPIR(
            tracker_state["params"],
            tracker_state["state"],
            tapir_kwargs={
                "bilinear_interp_with_depthwise_conv": False,
                "pyramid_level": 1,
                "extra_convs": True,
                "softmax_temperature": 10.0,
            },
        )
        autoencoder_params = _recover_tree(_load_npz(autoencoder_checkpoint))
        autoencoder = track_autoencoder.TrackAutoEncoder(
            decoder_scan_chunk_size=decoder_chunk_size
        )

        @jax.jit
        def track_chunk(video: Any, feature_grids: Any, query_points: Any) -> tuple[Any, Any]:
            outputs = tracker(
                video=video,
                query_points=query_points[None],
                is_training=False,
                query_chunk_size=tracker_chunk_size,
                feature_grids=feature_grids,
            )
            visible = model_utils.postprocess_occlusions(
                outputs["occlusion"], outputs["expected_dist"]
            )
            return outputs["tracks"][0], visible[0]

        @jax.jit
        def reconstruct(inputs: dict[str, Any]) -> Any:
            return autoencoder.apply({"params": autoencoder_params}, inputs)

    except MetricLoadError:
        raise
    except Exception as exc:  # pragma: no cover - checkpoint/runtime specific
        raise MetricLoadError(
            f"Failed to initialize official TRAJAN models: {type(exc).__name__}: {exc}"
        ) from exc

    loaded_runtime = {
        "jax": jax,
        "tracker": tracker,
        "model_utils": model_utils,
        "track_chunk": track_chunk,
        "reconstruct": reconstruct,
        "repo_root": repo_root,
        "tracker_checkpoint": tracker_checkpoint,
        "autoencoder_checkpoint": autoencoder_checkpoint,
        "tracker_chunk_size": tracker_chunk_size,
        "decoder_chunk_size": decoder_chunk_size,
    }
    _RUNTIME_CACHE[cache_key] = loaded_runtime
    return loaded_runtime


def _decode_video(path: Path, *, max_frames: int, analysis_size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (analysis_size, analysis_size):
            frame = cv2.resize(
                frame,
                (analysis_size, analysis_size),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    capture.release()
    if len(frames) < 2:
        raise ValueError(f"TRAJAN requires at least two decodable video frames: {path}")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def _sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _sample_query_points(
    rng: np.random.Generator,
    *,
    frame_count: int,
    height: int,
    width: int,
    point_count: int,
) -> np.ndarray:
    time = rng.integers(0, frame_count, size=(point_count, 1))
    y = rng.integers(0, height, size=(point_count, 1))
    x = rng.integers(0, width, size=(point_count, 1))
    return np.concatenate([time, y, x], axis=-1).astype(np.float32)


def _track_points(
    frames: np.ndarray,
    query_points: np.ndarray,
    loaded_runtime: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    model_utils = loaded_runtime["model_utils"]
    video = model_utils.preprocess_frames(frames[None])
    feature_grids = loaded_runtime["tracker"].get_feature_grids(
        video,
        is_training=False,
    )
    chunk_size = loaded_runtime["tracker_chunk_size"]
    all_tracks: list[np.ndarray] = []
    all_visible: list[np.ndarray] = []
    for start in range(0, len(query_points), chunk_size):
        tracks, visible = loaded_runtime["track_chunk"](
            video,
            feature_grids,
            query_points[start : start + chunk_size],
        )
        all_tracks.append(np.asarray(tracks))
        all_visible.append(np.asarray(visible, dtype=bool))
    return np.concatenate(all_tracks, axis=0), np.concatenate(all_visible, axis=0)


def _prepare_autoencoder_inputs(
    tracks: np.ndarray,
    visible: np.ndarray,
    *,
    rng: np.random.Generator,
    support_count: int,
    target_count: int,
    episode_length: int,
    analysis_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    frame_count = min(tracks.shape[1], episode_length)
    normalized_tracks = (tracks[:, :frame_count].astype(np.float32) + 0.5) / float(
        analysis_size
    )
    visible = visible[:, :frame_count]

    tracks_tq = np.transpose(normalized_tracks, (1, 0, 2))
    visible_tq = np.transpose(visible, (1, 0))
    pad_frames = episode_length - frame_count
    tracks_tq = np.pad(tracks_tq, ((0, pad_frames), (0, 0), (0, 0)))
    visible_tq = np.pad(visible_tq, ((0, pad_frames), (0, 0)))

    required = support_count + target_count
    if tracks_tq.shape[1] < required:
        raise ValueError(
            f"TRAJAN needs at least {required} tracked points, got {tracks_tq.shape[1]}"
        )
    indices = rng.permutation(tracks_tq.shape[1])
    target_indices = indices[:target_count]
    support_indices = indices[-support_count:]

    support_tracks = np.transpose(tracks_tq[:, support_indices], (1, 0, 2))
    support_visible = np.transpose(visible_tq[:, support_indices], (1, 0))[..., None]
    target_tracks = np.transpose(tracks_tq[:, target_indices], (1, 0, 2))
    target_visible = np.transpose(visible_tq[:, target_indices], (1, 0))

    query_frames = np.zeros(target_count, dtype=np.int32)
    for index in range(target_count):
        candidates = np.flatnonzero(target_visible[index, :frame_count])
        if candidates.size:
            query_frames[index] = int(rng.choice(candidates))
    query_xy = target_tracks[np.arange(target_count), query_frames]
    query_points = np.concatenate(
        [query_frames[:, None].astype(np.float32), query_xy],
        axis=1,
    )
    inputs = {
        "support_tracks": support_tracks[None].astype(np.float32),
        "support_tracks_visible": support_visible[None].astype(np.float32),
        "query_points": query_points[None].astype(np.float32),
        "boundary_frame": np.asarray([frame_count], dtype=np.int32),
    }
    return inputs, target_tracks, target_visible, query_frames


def compute_average_jaccard(
    *,
    ground_truth_tracks: np.ndarray,
    ground_truth_visible: np.ndarray,
    predicted_tracks: np.ndarray,
    predicted_visible: np.ndarray,
    query_frames: np.ndarray,
    thresholds: tuple[int, ...] = AJ_THRESHOLDS,
) -> dict[str, Any]:
    """Compute TAP-Vid Average Jaccard using the official threshold protocol."""
    if ground_truth_tracks.shape != predicted_tracks.shape:
        raise ValueError("TRAJAN ground-truth and predicted track shapes must match")
    if ground_truth_visible.shape != predicted_visible.shape:
        raise ValueError("TRAJAN visibility shapes must match")
    point_count, frame_count = ground_truth_visible.shape
    evaluation_points = np.ones((point_count, frame_count), dtype=bool)
    valid_queries = np.clip(np.rint(query_frames).astype(np.int64), 0, frame_count - 1)
    evaluation_points[np.arange(point_count), valid_queries] = False

    gt_visible = ground_truth_visible.astype(bool)
    pred_visible = predicted_visible.astype(bool)
    squared_distance = np.sum(
        np.square(predicted_tracks - ground_truth_tracks),
        axis=-1,
    )
    per_threshold: dict[str, float] = {}
    for threshold in thresholds:
        within = squared_distance < float(threshold * threshold)
        correct = within & gt_visible
        true_positives = np.sum(correct & pred_visible & evaluation_points)
        ground_truth_positives = np.sum(gt_visible & evaluation_points)
        false_positives = np.sum(
            ((~gt_visible & pred_visible) | (~within & pred_visible))
            & evaluation_points
        )
        denominator = int(ground_truth_positives + false_positives)
        per_threshold[str(threshold)] = (
            float(true_positives) / float(denominator) if denominator else 0.0
        )

    evaluated = int(np.sum(evaluation_points))
    occlusion_accuracy = (
        float(np.sum((gt_visible == pred_visible) & evaluation_points)) / float(evaluated)
        if evaluated
        else 0.0
    )
    return {
        "average_jaccard": float(np.mean(list(per_threshold.values()))),
        "jaccard_by_threshold": per_threshold,
        "occlusion_accuracy": occlusion_accuracy,
    }


def compute_trajan_average_jaccard(
    *,
    video_path: Path,
    sample_id: str,
    normalization: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Score one generated video with the official TRAJAN reconstruction protocol."""
    from worldatlas.benchmark.metrics.base import normalize_score

    analysis_size = _as_positive_int(runtime.get("analysis_size"), 256)
    if analysis_size != 256:
        raise ValueError("TRAJAN official protocol requires analysis_size=256")
    episode_length = min(_as_positive_int(runtime.get("max_frames"), 150), 150)
    support_count = _as_positive_int(runtime.get("support_track_count"), 2048)
    target_count = _as_positive_int(runtime.get("target_track_count"), 2048)
    loaded_runtime = _load_runtime(runtime)
    decoder_chunk_size = loaded_runtime["decoder_chunk_size"]
    if target_count % decoder_chunk_size:
        raise ValueError(
            "TRAJAN target_track_count must be divisible by decoder_chunk_size; "
            f"got {target_count} and {decoder_chunk_size}"
        )

    frames = _decode_video(
        Path(video_path),
        max_frames=episode_length,
        analysis_size=analysis_size,
    )
    seed = _sample_seed(int(runtime.get("seed", 17)), sample_id)
    rng = np.random.default_rng(seed)
    point_count = support_count + target_count
    query_points = _sample_query_points(
        rng,
        frame_count=len(frames),
        height=analysis_size,
        width=analysis_size,
        point_count=point_count,
    )
    tracks, visible = _track_points(frames, query_points, loaded_runtime)
    inputs, target_tracks, target_visible, query_frames = _prepare_autoencoder_inputs(
        tracks,
        visible,
        rng=rng,
        support_count=support_count,
        target_count=target_count,
        episode_length=150,
        analysis_size=analysis_size,
    )
    outputs = loaded_runtime["reconstruct"](inputs)
    predicted_tracks = np.asarray(outputs.tracks[0], dtype=np.float32)
    predicted_occluded = np.asarray(
        loaded_runtime["model_utils"].postprocess_occlusions(
            outputs.visible_logits,
            outputs.certain_logits,
        )[0, ..., 0],
        dtype=bool,
    )

    frame_count = len(frames)
    evaluation_query_frames = np.zeros_like(query_frames)
    score = compute_average_jaccard(
        ground_truth_tracks=target_tracks[:, :frame_count] * analysis_size,
        ground_truth_visible=target_visible[:, :frame_count],
        predicted_tracks=predicted_tracks[:, :frame_count] * analysis_size,
        predicted_visible=~predicted_occluded[:, :frame_count],
        query_frames=evaluation_query_frames,
    )
    raw = float(score["average_jaccard"])
    if not np.isfinite(raw):
        raise ValueError(f"TRAJAN returned a non-finite Average Jaccard: {raw}")
    return {
        "raw": raw,
        "normalized": normalize_score(raw, normalization),
        "backend": "trajan",
        "details": {
            "video_path": str(video_path),
            "repo_root": (
                str(loaded_runtime["repo_root"])
                if loaded_runtime["repo_root"] is not None
                else None
            ),
            "tracker_checkpoint": str(loaded_runtime["tracker_checkpoint"]),
            "autoencoder_checkpoint": str(loaded_runtime["autoencoder_checkpoint"]),
            "tracker_model_type": "bootstapir",
            "frame_count": frame_count,
            "episode_length": 150,
            "analysis_size": analysis_size,
            "sampled_track_count": point_count,
            "support_track_count": support_count,
            "target_track_count": target_count,
            "thresholds_px": list(AJ_THRESHOLDS),
            "evaluation_query_mode": "strided_from_first_frame",
            "jaccard_by_threshold": score["jaccard_by_threshold"],
            "occlusion_accuracy": score["occlusion_accuracy"],
            "seed": seed,
            "raw_direction": "higher_is_better",
        },
    }


__all__ = ["AJ_THRESHOLDS", "compute_average_jaccard", "compute_trajan_average_jaccard"]
