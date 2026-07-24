"""Small shared helpers for in-tree official metric backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from worldarena.common.checkpoints import checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def thirdparty_path(repo: str, *parts: str) -> str:
    """Return an absolute path inside a configured third-party checkout."""
    return str((PROJECT_ROOT / "thirdparty" / repo).joinpath(*parts).resolve())


def official_checkpoint_path(*parts: str, required: bool = True) -> str:
    """Resolve a checkpoint used by the official metric implementations."""
    return str(
        checkpoint_path(
            "official_metrics",
            *parts,
            kind="any",
            required=required,
        )
    )


class BaseMetric:
    """Minimal compatibility base used by the imported metric algorithms."""

    def __init__(self) -> None:
        import torch

        self._metric: Any = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _process_np_to_tensor(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> tuple[Any, Any]:
        from torchvision.transforms import ToTensor

        img1 = ToTensor()(rendered_image).unsqueeze(0).to(self._device)
        img2 = ToTensor()(reference_image).unsqueeze(0).to(self._device)
        return img1, img2


__all__ = ["BaseMetric", "official_checkpoint_path", "thirdparty_path"]
