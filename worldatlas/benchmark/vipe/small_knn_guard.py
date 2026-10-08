"""Input-only guard for official ViPE's zero-capacity CUDA KD-tree.

ViPE 95a8816 allocates (N // 64) * 16 nodes, then writes node zero even
when N < 64. Pad only that case with remote points; all returned neighbours
must still belong to the original tree. NVIDIA's extension/weights stay intact.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path

import torch

POLICY = "official_native_knn_small_tree_remote_padding_v1"
MIN_TREE_SIZE = 64
GUARD_EVENTS: ContextVar[list | None] = ContextVar("vipe_knn_guard_events", default=None)


def guarded_nearest_neighbours(native, query, tree, k, record=None):
    n = tree.shape[0]
    if n >= MIN_TREE_SIZE or n < k or k <= 0:
        return native(query, tree, k)
    if query.ndim != 2 or tree.ndim != 2 or query.shape[1] != tree.shape[1] or tree.shape[1] not in (2, 3):
        return native(query, tree, k)
    if query.shape[0] == 0:
        return (query.new_empty((0, k)), torch.empty((0, k), device=tree.device, dtype=torch.int32))
    if not torch.isfinite(query).all() or not torch.isfinite(tree).all():
        raise ValueError("Official ViPE KNN requires finite coordinates")
    bound = max(1.0, float(query.abs().max().item()), float(tree.abs().max().item()))
    # Real pair distances <= 2*bound*sqrt(D); dummy distances >= 15*bound*sqrt(D).
    # Thus no dummy can be among the k nearest when k <= the number of real points.
    far = 16.0 * bound + 1.0
    padding = tree.new_full((MIN_TREE_SIZE - n, tree.shape[1]), far)
    if not torch.isfinite(padding).all():
        raise ValueError("Coordinates too large to safely pad official ViPE KNN")
    distances, indices = native(query, torch.cat((tree, padding), dim=0).contiguous(), k)
    if not bool(((indices >= 0) & (indices < n)).all()):
        raise RuntimeError("Official ViPE KNN returned a padding point; refusing altered neighbours")
    if record is not None:
        record({"policy": POLICY, "tree_points": n, "padded_tree_points": MIN_TREE_SIZE,
                "query_points": query.shape[0], "k": k, "coordinate_bound": bound})
    return distances, indices


def install(log_path: Path | None = None):
    """Install once; record in the active video's context and optional retry log."""
    import vipe_ext
    native = vipe_ext.utils_ext.nearest_neighbours
    if getattr(native, "_worldatlas_small_tree_guard", False):
        return

    def record(event):
        events = GUARD_EVENTS.get()
        if events is not None:
            events.append(event)
        if log_path is not None:
            with log_path.open("a") as stream:
                stream.write(json.dumps(event) + "\n")

    def guarded(query, tree, k):
        return guarded_nearest_neighbours(native, query, tree, k, record)

    guarded._worldatlas_small_tree_guard = True
    vipe_ext.utils_ext.nearest_neighbours = guarded
