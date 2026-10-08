"""Load memory-track Prior-Depth-Anything from WorldAtlas ``thirdparty/``.

The official constructor takes ``mde_dir`` / ``ckpt_dir``. Local Hope weights
are the v1.0 names (``prior_depth_anything_vitb.pth``), so this wrapper pins
``version='1.0'`` and does not import NVIDIA ViPE's vendored ``priorda``.

Official ``prior_depth_anything`` imports ``torch_cluster`` at module load.
Hope images do not ship that wheel. This wrapper installs a batched-cdist
``knn`` shim with the same ``[2, M*K]`` layout before importing the clone.

Grid bucketing was measured against this on the real workload and lost: the
SLAM projection supplies only ~4.6k sparse points against ~897k queries, so
ring expansion degenerates while the brute-force scan stays GPU-friendly.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.common.checkpoints import project_root
from worldatlas.benchmark.vipe.depth_contract import pinhole_supported_camera_types
from worldatlas.benchmark.vipe.memory_weights import priorda_dir

PRIORDA_REPO = project_root() / "thirdparty" / "Prior-Depth-Anything"
_KNN_CHUNK = 65536
_SHIM_FLAG = "_worldatlas_torch_cluster_shim"


def official_priorda_root() -> Path:
    explicit = os.environ.get("WORLDATLAS_PRIORDA_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return PRIORDA_REPO


def torch_cluster_knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
    cosine: bool = False,
    num_workers: int = 1,
) -> torch.Tensor:
    """``torch_cluster.knn`` layout: ``[2, M*k]`` (query index, neighbor index).

    Official PriorDA does ``knn_map[1].view(-1, k)`` as indices into ``x``.
    Hope has no ``torch_cluster`` wheel; this matches a batched
    ``cdist`` fallback, including padding when a batch has fewer than ``k`` points.
    """
    del num_workers
    if cosine:
        raise NotImplementedError("Prior-Depth-Anything does not use cosine knn")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"knn expects 2D feature matrices, got x={tuple(x.shape)} y={tuple(y.shape)}")
    if batch_x is None:
        batch_x = x.new_zeros(x.shape[0], dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.shape[0], dtype=torch.long)
    batch_x = batch_x.to(device=x.device, dtype=torch.long)
    batch_y = batch_y.to(device=y.device, dtype=torch.long)

    neighbors = y.new_empty((y.shape[0], int(k)), dtype=torch.long)
    filled = y.new_zeros((y.shape[0],), dtype=torch.bool)
    for batch_id in batch_y.unique(sorted=True):
        x_mask = batch_x == batch_id
        y_mask = batch_y == batch_id
        x_batch = x[x_mask]
        y_batch = y[y_mask]
        if y_batch.numel() == 0:
            continue
        if x_batch.numel() == 0:
            raise ValueError(f"PriorDA knn shim found no reference points in batch {int(batch_id)}.")
        global_x = torch.nonzero(x_mask, as_tuple=False).flatten()
        global_y = torch.nonzero(y_mask, as_tuple=False).flatten()
        local_k = min(int(k), x_batch.shape[0])
        offset = 0
        for y_chunk in torch.split(y_batch, _KNN_CHUNK):
            dists = torch.cdist(y_chunk.float(), x_batch.float())
            local = torch.topk(dists, k=local_k, dim=1, largest=False).indices
            chosen = global_x[local]
            if local_k < int(k):
                pad = chosen[:, -1:].expand(-1, int(k) - local_k)
                chosen = torch.cat([chosen, pad], dim=1)
            sl = global_y[offset : offset + y_chunk.shape[0]]
            neighbors[sl] = chosen
            filled[sl] = True
            offset += y_chunk.shape[0]
    if not bool(filled.all()):
        raise ValueError("PriorDA knn shim left unfilled query points.")
    query = torch.arange(y.shape[0], device=y.device).unsqueeze(1).expand(-1, int(k))
    return torch.stack([query.reshape(-1), neighbors.reshape(-1)], dim=0)


def ensure_torch_cluster_shim() -> types.ModuleType:
    """Provide ``torch_cluster.knn`` when the real wheel is absent."""
    existing = sys.modules.get("torch_cluster")
    if existing is not None and callable(getattr(existing, "knn", None)):
        return existing
    try:
        import torch_cluster as installed
    except ImportError:
        installed = None
    if installed is not None and callable(getattr(installed, "knn", None)):
        return installed

    module = types.ModuleType("torch_cluster")
    module.knn = torch_cluster_knn
    setattr(module, _SHIM_FLAG, True)
    sys.modules["torch_cluster"] = module
    return module


def ensure_priorda_on_path() -> Path:
    ensure_torch_cluster_shim()
    root = official_priorda_root()
    package = root / "prior_depth_anything"
    if not package.is_dir():
        raise MetricLoadError(
            f"Official Prior-Depth-Anything is missing at {root}. Clone "
            "https://github.com/SpatialVision/Prior-Depth-Anything into "
            "WorldAtlas/thirdparty/Prior-Depth-Anything."
        )
    inserted = str(root)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    return root


class ArenaPriorDAModel:
    """Official-ViPE PriorDA wrapper backed by the official clone."""

    def __init__(self, weights_dir: str | None = None, device: str = "cuda") -> None:
        ensure_priorda_on_path()
        from prior_depth_anything import PriorDepthAnything

        weights = Path(weights_dir).expanduser() if weights_dir else priorda_dir()
        dav2 = weights / "depth_anything_v2_vitb.pth"
        ckpt = weights / "prior_depth_anything_vitb.pth"
        if not dav2.is_file() or not ckpt.is_file():
            raise MetricLoadError(
                f"Prior-Depth-Anything weights missing under {weights}: "
                "need depth_anything_v2_vitb.pth and prior_depth_anything_vitb.pth"
            )
        # v1.1 looks for prior_depth_anything_vitb_1_1.pth; Hope ships the v1.0 name.
        self.model = PriorDepthAnything(
            device=device,
            version="1.0",
            mde_dir=str(weights),
            ckpt_dir=str(weights),
        )

    @property
    def depth_type(self):
        from vipe.priors.depth.base import DepthType

        return DepthType.METRIC_DEPTH

    @property
    def supported_camera_types(self):
        return pinhole_supported_camera_types()

    def estimate(self, src, *, pattern: str | None = None):
        from vipe.priors.depth.base import DepthEstimationResult
        from vipe.utils.misc import unpack_optional

        rgb = unpack_optional(src.rgb)
        prompt_metric_depth = unpack_optional(src.prompt_metric_depth)
        assert rgb.dim() == 3 and prompt_metric_depth.dim() == 2, "Single batch only"
        final_depth = self.model.infer_one_sample(
            image=rgb * 255.0,
            prior=prompt_metric_depth,
            geometric=None,
            pattern=pattern,
        )
        return DepthEstimationResult(metric_depth=final_depth)
