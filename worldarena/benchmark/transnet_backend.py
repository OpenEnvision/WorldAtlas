"""TransNetV2 backend for segment-continuity evaluation."""

from __future__ import annotations

from functools import lru_cache
import importlib
import sys
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from worldarena.common.checkpoints import checkpoint_path, project_root

TRANSNET_INPUT_SIZE = (27, 48, 3)
DEFAULT_TRANSNET_REPO_CANDIDATES = (
    project_root() / "thirdparty" / "TransNetV2" / "inference",
    project_root() / "third_party" / "TransNetV2" / "inference",
)
DEFAULT_TRANSNET_PYTORCH_REPO_CANDIDATES = (
    project_root() / "thirdparty" / "TransNetV2" / "inference-pytorch",
    project_root() / "third_party" / "TransNetV2" / "inference-pytorch",
)


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize an RGB frame to TransNetV2 input resolution."""
    height, width = TRANSNET_INPUT_SIZE[0], TRANSNET_INPUT_SIZE[1]
    if frame.shape[0] == height and frame.shape[1] == width:
        return np.asarray(frame, dtype=np.uint8)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def prepare_transnet_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Convert decoded RGB frames into TransNetV2 `[N, 27, 48, 3]` input."""
    if not frames:
        raise ValueError("no frames available for TransNetV2 inference")
    return np.stack([_resize_frame(frame) for frame in frames], axis=0).astype(np.uint8, copy=False)


def _resolve_transnet_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.exists():
            raise FileNotFoundError(f"TransNetV2 repo_root does not exist: {repo_root}")
        return repo_root.resolve()

    for candidate in DEFAULT_TRANSNET_REPO_CANDIDATES:
        if (candidate / "transnetv2.py").is_file():
            return candidate.resolve()
    return None


def _resolve_transnet_weights(runtime: dict[str, Any]) -> Path:
    configured = runtime.get("weights_dir") or runtime.get("model_dir")
    if configured:
        weights_dir = Path(str(configured)).expanduser()
        if not weights_dir.is_dir():
            raise FileNotFoundError(f"TransNetV2 weights_dir does not exist: {weights_dir}")
        return weights_dir.resolve()

    repo_root = _resolve_transnet_repo_root(runtime)
    if repo_root is not None:
        bundled = repo_root / "transnetv2-weights"
        if bundled.is_dir():
            return bundled.resolve()

    optional = checkpoint_path("TransNetV2", "transnetv2-weights", kind="dir", required=False)
    if optional.is_dir():
        return optional.resolve()

    raise FileNotFoundError(
        "TransNetV2 weights are unavailable; clone thirdparty/TransNetV2 or place "
        "transnetv2-weights under the checkpoint root."
    )


def _load_transnet_class(runtime: dict[str, Any]) -> tuple[Any, Path, Path]:
    repo_root = _resolve_transnet_repo_root(runtime)
    if repo_root is None:
        raise ModuleNotFoundError(
            "TransNetV2 inference code is unavailable; clone "
            "https://github.com/soCzech/TransNetV2 into thirdparty/TransNetV2 "
            "or set benchmark.metric_runtime.segment_continuity.repo_root."
        )

    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        module = importlib.import_module("transnetv2")
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError(
            "TransNetV2 package import failed; install tensorflow and ensure "
            f"thirdparty/TransNetV2/inference is available. {exc.__class__.__name__}: {exc}"
        ) from exc

    transnet_class = getattr(module, "TransNetV2", None)
    if transnet_class is None:
        raise AttributeError("TransNetV2 inference module does not expose TransNetV2")

    weights_dir = _resolve_transnet_weights(runtime)
    return transnet_class, repo_root, weights_dir


@lru_cache(maxsize=16)
def _cached_transnet_model(transnet_class: Any, weights_dir: str) -> Any:
    """Load one TransNetV2 model per worker/weights pair."""
    return transnet_class(weights_dir)


def _transnet_model(runtime: dict[str, Any]) -> tuple[Any, Path, Path]:
    transnet_class, repo_root, weights_dir = _load_transnet_class(runtime)
    if bool(runtime.get("cache_instance", True)):
        model = _cached_transnet_model(transnet_class, str(weights_dir))
    else:
        model = transnet_class(str(weights_dir))
    return model, repo_root, weights_dir


def _predict_prepared_frames_tensorflow(video_frames: np.ndarray, runtime: dict[str, Any]) -> dict[str, Any]:
    model, repo_root, weights_dir = _transnet_model(runtime)
    single_frame_predictions, _all_frame_predictions = model.predict_frames(video_frames)
    probabilities = np.asarray(single_frame_predictions, dtype=np.float32).reshape(-1)
    return {
        "probabilities": probabilities,
        "repo_root": str(repo_root),
        "weights_dir": str(weights_dir),
        "frame_count": len(video_frames),
        "inference_framework": "tensorflow",
        "inference_device": "tensorflow_default",
    }


def _resolve_transnet_pytorch_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("pytorch_repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not (repo_root / "transnetv2_pytorch.py").is_file():
            raise FileNotFoundError(f"TransNetV2 PyTorch repo_root is invalid: {repo_root}")
        return repo_root.resolve()

    for candidate in DEFAULT_TRANSNET_PYTORCH_REPO_CANDIDATES:
        if (candidate / "transnetv2_pytorch.py").is_file():
            return candidate.resolve()
    return None


def _resolve_transnet_pytorch_weights(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("pytorch_weights")
    if configured:
        weights = Path(str(configured)).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"TransNetV2 PyTorch weights do not exist: {weights}")
        return weights.resolve()

    candidates = (
        checkpoint_path("TransNetV2", "transnetv2-pytorch-weights.pth", kind="file", required=False),
        project_root().parent / "ckpt" / "TransNetV2" / "transnetv2-pytorch-weights.pth",
        project_root() / "thirdparty" / "TransNetV2" / "inference-pytorch" / "transnetv2-pytorch-weights.pth",
        project_root() / "third_party" / "TransNetV2" / "inference-pytorch" / "transnetv2-pytorch-weights.pth",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_transnet_pytorch_class(runtime: dict[str, Any]) -> tuple[Any, Path, Path]:
    repo_root = _resolve_transnet_pytorch_repo_root(runtime)
    if repo_root is None:
        raise ModuleNotFoundError("TransNetV2 PyTorch inference code is unavailable")
    weights = _resolve_transnet_pytorch_weights(runtime)
    if weights is None:
        raise FileNotFoundError("TransNetV2 PyTorch weights are unavailable")

    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    module = importlib.import_module("transnetv2_pytorch")
    transnet_class = getattr(module, "TransNetV2", None)
    if transnet_class is None:
        raise AttributeError("TransNetV2 PyTorch module does not expose TransNetV2")
    return transnet_class, repo_root, weights


@lru_cache(maxsize=16)
def _cached_transnet_pytorch_model(transnet_class: Any, weights_file: str, device: str) -> Any:
    """Load one converted TransNetV2 model per worker/device."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before parallel work starts.
        pass
    model = transnet_class()
    try:
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(torch.device(device))
    return model


def _transnet_pytorch_model(runtime: dict[str, Any]) -> tuple[Any, Path, Path, str]:
    import torch

    transnet_class, repo_root, weights_file = _load_transnet_pytorch_class(runtime)
    device = str(runtime.get("pytorch_device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("TransNetV2 PyTorch CUDA inference requested but CUDA is unavailable")
    if bool(runtime.get("cache_instance", True)):
        model = _cached_transnet_pytorch_model(transnet_class, str(weights_file), device)
    else:
        model = _cached_transnet_pytorch_model.__wrapped__(transnet_class, str(weights_file), device)
    return model, repo_root, weights_file, device


def _transnet_input_windows(video_frames: np.ndarray):
    """Yield the exact 100-frame/50-stride windows used by TransNetV2."""
    frame_count = len(video_frames)
    no_padded_frames_end = 25 + 50 - (frame_count % 50 if frame_count % 50 != 0 else 50)
    padded_inputs = np.concatenate(
        [
            np.repeat(video_frames[:1], 25, axis=0),
            video_frames,
            np.repeat(video_frames[-1:], no_padded_frames_end, axis=0),
        ],
        axis=0,
    )
    for pointer in range(0, len(padded_inputs) - 99, 50):
        yield padded_inputs[pointer : pointer + 100]


def _predict_prepared_frames_pytorch(video_frames: np.ndarray, runtime: dict[str, Any]) -> dict[str, Any]:
    import torch

    model, repo_root, weights_file, device = _transnet_pytorch_model(runtime)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for window in _transnet_input_windows(video_frames):
            inputs = torch.from_numpy(np.ascontiguousarray(window[None])).to(torch.device(device))
            logits, _all_frame_logits = model(inputs)
            predictions.append(torch.sigmoid(logits)[0, 25:75, 0].cpu().numpy())
    probabilities = np.concatenate(predictions)[: len(video_frames)].astype(np.float32, copy=False)
    return {
        "probabilities": probabilities,
        "repo_root": str(repo_root),
        "weights_dir": str(weights_file.parent),
        "weights_file": str(weights_file),
        "frame_count": len(video_frames),
        "inference_framework": "pytorch",
        "inference_device": device,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _pytorch_cuda_available(runtime: dict[str, Any]) -> bool:
    if _resolve_transnet_pytorch_repo_root(runtime) is None:
        return False
    if _resolve_transnet_pytorch_weights(runtime) is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _predict_prepared_frames(video_frames: np.ndarray, runtime: dict[str, Any]) -> dict[str, Any]:
    framework = str(runtime.get("inference_framework", "auto")).strip().lower()
    if framework not in {"auto", "pytorch", "tensorflow"}:
        raise ValueError(f"unsupported TransNetV2 inference_framework: {framework}")
    if framework == "pytorch" or (framework == "auto" and _pytorch_cuda_available(runtime)):
        return _predict_prepared_frames_pytorch(video_frames, runtime)
    return _predict_prepared_frames_tensorflow(video_frames, runtime)


def _ratio_window(frame_count: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, -1
    start = min(max(float(start_ratio), 0.0), 1.0)
    end = min(max(float(end_ratio), 0.0), 1.0)
    if end < start:
        end = start
    start_index = int(round(start * max(frame_count - 1, 0)))
    end_index = int(round(end * max(frame_count - 1, 0)))
    return start_index, max(start_index, end_index)


def _decode_transnet_video_frames(
    path: Path,
    *,
    start_ratio: float,
    end_ratio: float,
) -> tuple[np.ndarray, int, int, int]:
    """Decode the native evaluation window directly at TransNetV2 resolution."""
    capture = None
    for attempt in range(5):
        candidate = cv2.VideoCapture(str(path))
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()
        if attempt < 4:
            time.sleep(0.2 * (attempt + 1))
    if capture is None:
        raise RuntimeError(f"failed to open video for TransNetV2: {path}")

    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_count > 0:
        start_index, end_index = _ratio_window(reported_count, start_ratio, end_ratio)
        frames: list[np.ndarray] = []
        decoded_count = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if start_index <= decoded_count <= end_index:
                resized = cv2.resize(
                    frame,
                    (TRANSNET_INPUT_SIZE[1], TRANSNET_INPUT_SIZE[0]),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            decoded_count += 1
            if decoded_count > end_index:
                break
        native_frame_count = max(reported_count, decoded_count)
    else:
        decoded: list[np.ndarray] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            resized = cv2.resize(
                frame,
                (TRANSNET_INPUT_SIZE[1], TRANSNET_INPUT_SIZE[0]),
                interpolation=cv2.INTER_AREA,
            )
            decoded.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        native_frame_count = len(decoded)
        start_index, end_index = _ratio_window(native_frame_count, start_ratio, end_ratio)
        frames = decoded[start_index : end_index + 1]
    capture.release()

    if not frames:
        raise RuntimeError(f"video contains no decodable frames for TransNetV2: {path}")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False), native_frame_count, start_index, end_index


def predict_transnet_single_frame_probabilities(
    frames: list[np.ndarray],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run TransNetV2 and return per-frame single-head boundary probabilities."""
    runtime = dict(runtime or {})
    video_frames = prepare_transnet_frames(frames)
    result = _predict_prepared_frames(video_frames, runtime)
    result["analysis_frame_policy"] = "loaded_sampled_frames"
    return result


def predict_transnet_video_probabilities(
    path: str | Path,
    *,
    runtime: dict[str, Any] | None = None,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> dict[str, Any]:
    """Run TransNetV2 on every native frame in the selected video window."""
    runtime = dict(runtime or {})
    video_path = Path(path)
    video_frames, native_frame_count, start_index, end_index = _decode_transnet_video_frames(
        video_path,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
    )
    result = _predict_prepared_frames(video_frames, runtime)
    result.update(
        {
            "analysis_frame_policy": "prediction_native_window",
            "native_frame_count": native_frame_count,
            "native_window_start_index": start_index,
            "native_window_end_index": end_index,
        }
    )
    return result


__all__ = [
    "TRANSNET_INPUT_SIZE",
    "predict_transnet_single_frame_probabilities",
    "predict_transnet_video_probabilities",
    "prepare_transnet_frames",
]
