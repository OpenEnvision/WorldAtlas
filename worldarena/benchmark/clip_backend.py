from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from worldarena.common.checkpoints import hf_local_dir


DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
ALLOWED_CLIP_STATE_DICT_UNEXPECTED_KEYS = {
    "text_model.embeddings.position_ids",
    "vision_model.embeddings.position_ids",
}
_REQUIRED_CLIP_CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
_CLIP_TOKENIZER_FILE_GROUPS = (
    ("tokenizer.json",),
    ("vocab.json", "merges.txt"),
)


@dataclass(frozen=True)
class CachedClipArtifacts:
    config_dir: Path
    weights_path: Path


def _has_clip_processor_files(model_dir: Path) -> bool:
    if not all((model_dir / filename).is_file() for filename in _REQUIRED_CLIP_CONFIG_FILES):
        return False
    return any(
        all((model_dir / filename).is_file() for filename in tokenizer_group)
        for tokenizer_group in _CLIP_TOKENIZER_FILE_GROUPS
    )


def _candidate_clip_weight_paths(model_dir: Path) -> list[Path]:
    candidates = [model_dir / "model.safetensors"]
    if model_dir.parent.name == "snapshots":
        candidates.extend(
            snapshot_dir / "model.safetensors"
            for snapshot_dir in sorted(model_dir.parent.iterdir())
            if snapshot_dir.is_dir() and snapshot_dir != model_dir
        )
    candidates.append(model_dir / "pytorch_model.bin")
    return candidates


def _resolve_clip_weights_path(model_dir: Path) -> Path:
    for candidate in _candidate_clip_weight_paths(model_dir):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"CLIP weights not found under {model_dir}. Expected model.safetensors or pytorch_model.bin."
    )


@lru_cache(maxsize=4)
def resolve_cached_clip_artifacts(
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> CachedClipArtifacts:
    model_dir = hf_local_dir(model_name, required=True)
    if not _has_clip_processor_files(model_dir):
        raise FileNotFoundError(
            "CLIP processor files are incomplete under "
            f"{model_dir}. Expected {_REQUIRED_CLIP_CONFIG_FILES} and tokenizer files."
        )
    weights_path = _resolve_clip_weights_path(model_dir)
    return CachedClipArtifacts(config_dir=model_dir, weights_path=weights_path)


@lru_cache(maxsize=4)
def load_clip_backend(
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> tuple[str, Any, Any, str, str | None, str | None] | None:
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoProcessor, CLIPConfig, CLIPModel
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = resolve_cached_clip_artifacts(model_name)
    if artifacts is not None:
        processor = AutoProcessor.from_pretrained(str(artifacts.config_dir))
        if artifacts.weights_path.name == "model.safetensors":
            config = CLIPConfig.from_pretrained(str(artifacts.config_dir))
            model = CLIPModel(config)
            state_dict = load_file(str(artifacts.weights_path))
            incompatible = model.load_state_dict(state_dict, strict=False)
            unexpected_keys = set(incompatible.unexpected_keys)
            missing_keys = set(incompatible.missing_keys)
            if missing_keys or not unexpected_keys.issubset(ALLOWED_CLIP_STATE_DICT_UNEXPECTED_KEYS):
                raise RuntimeError(
                    "unexpected CLIP state_dict mismatch when loading cached safetensors: "
                    f"missing_keys={sorted(missing_keys)} unexpected_keys={sorted(unexpected_keys)}"
                )
            loader_name = "safetensors_manual"
        else:
            model = CLIPModel.from_pretrained(str(artifacts.config_dir))
            loader_name = "hf_local_bin"
        model = model.to(device)
        model.eval()
        return (
            loader_name,
            model,
            processor,
            device,
            str(artifacts.config_dir),
            str(artifacts.weights_path),
        )


def clip_backend_available(model_name: str = DEFAULT_CLIP_MODEL_NAME) -> bool:
    try:
        return load_clip_backend(model_name) is not None
    except FileNotFoundError:
        return False


def _clip_feature_tensor(output: Any) -> Any:
    return output.pooler_output if hasattr(output, "pooler_output") else output


def clip_image_features(
    frames: Sequence[np.ndarray],
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> tuple[np.ndarray, str]:
    backend = load_clip_backend(model_name)
    if backend is None:
        raise ModuleNotFoundError("transformers/torch/safetensors CLIP backend is unavailable")

    import torch
    import torch.nn.functional as F

    loader_name, model, processor, device, _, _ = backend
    inputs = processor(
        images=[Image.fromarray(frame) for frame in frames],
        return_tensors="pt",
        padding=True,
    )
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        features = _clip_feature_tensor(model.get_image_features(pixel_values=pixel_values))
        features = F.normalize(features, dim=-1)
    return features.detach().cpu().numpy(), loader_name


def clip_text_features(
    texts: Sequence[str],
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> tuple[np.ndarray, str]:
    backend = load_clip_backend(model_name)
    if backend is None:
        raise ModuleNotFoundError("transformers/torch/safetensors CLIP backend is unavailable")

    import torch
    import torch.nn.functional as F

    loader_name, model, processor, device, _, _ = backend
    max_position_embeddings = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
    processor_kwargs: dict[str, Any] = {
        "text": list(texts),
        "return_tensors": "pt",
        "padding": True,
    }
    if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = max_position_embeddings
    inputs = processor(**processor_kwargs)
    if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
        if "attention_mask" in inputs and inputs["attention_mask"].shape[-1] > max_position_embeddings:
            inputs["attention_mask"] = inputs["attention_mask"][..., :max_position_embeddings]
        if "input_ids" in inputs and inputs["input_ids"].shape[-1] > max_position_embeddings:
            inputs["input_ids"] = inputs["input_ids"][..., :max_position_embeddings]
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = _clip_feature_tensor(
            model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        )
        features = F.normalize(features, dim=-1)
    return features.detach().cpu().numpy(), loader_name
