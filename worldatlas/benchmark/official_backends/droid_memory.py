"""Release DROID CUDA state after each sample.

DROID-SLAM's ``DepthVideo`` calls ``tensor.share_memory_()`` so a visualizer
process can attach. Official eval never starts that process. On CUDA, those
calls create IPC handles that ``empty_cache`` cannot free. Echo 1280x704 /
241-frame clips then pin an 80 GiB GPU after a few dozen samples; a fresh
CUDA retry process OOMs on the first real clip (~77 GiB already allocated).

Even after no-op'ing ``share_memory_``, in-process eval still leaked ~1 GiB
per sample: ``DepthVideo.__del__`` and factor-graph correlation pyramids
participate in reference cycles, so cyclic GC never drops the live PyTorch
storage (OOM logs showed ~77 GiB still allocated, not cached). Drop tensor
attributes explicitly before releasing the Python objects. Scoring is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

_DEPTH_VIDEO_TENSORS = (
    "tstamp",
    "images",
    "dirty",
    "red",
    "poses",
    "disps",
    "disps_sens",
    "disps_up",
    "intrinsics",
    "fmaps",
    "nets",
    "inps",
)

_FACTOR_GRAPH_TENSORS = (
    "coords0",
    "ii",
    "jj",
    "age",
    "corr",
    "net",
    "inp",
    "damping",
    "target",
    "weight",
    "ii_inac",
    "jj_inac",
    "ii_bad",
    "jj_bad",
    "target_inac",
    "weight_inac",
)

_OWNER_CACHED_TENSORS = (
    "net",
    "inp",
    "fmap",
    "cnet",
    "fnet",
    "update",
    "update_op",
    "MEAN",
    "STDV",
    "last_pose",
    "last_disp",
    "last_time",
)


@contextmanager
def without_shared_cuda_memory() -> Iterator[None]:
    """No-op ``Tensor.share_memory_`` for in-process DROID construction and tracking."""
    import torch

    original = torch.Tensor.share_memory_

    def _keep_private(self):
        return self

    torch.Tensor.share_memory_ = _keep_private
    try:
        yield
    finally:
        torch.Tensor.share_memory_ = original


def _drop_attr(obj: Any, name: str) -> None:
    if obj is None or not hasattr(obj, name):
        return
    try:
        setattr(obj, name, None)
    except (AttributeError, TypeError):
        try:
            delattr(obj, name)
        except (AttributeError, TypeError):
            pass


def _clear_corr_pyramids(corr: Any) -> None:
    if corr is None:
        return
    for name in ("corr_pyramid", "pyramid"):
        pyramid = getattr(corr, name, None)
        if isinstance(pyramid, list):
            pyramid.clear()


def clear_factor_graph(graph: Any) -> None:
    """Drop factor-graph CUDA tensors. Does not change how scores were computed."""
    if graph is None:
        return
    _clear_corr_pyramids(getattr(graph, "corr", None))
    clear_edges = getattr(graph, "clear_edges", None)
    if callable(clear_edges):
        try:
            clear_edges()
        except Exception:
            pass
    for name in _FACTOR_GRAPH_TENSORS:
        _drop_attr(graph, name)
    _drop_attr(graph, "video")
    _drop_attr(graph, "update_op")


def _clear_depth_video(video: Any) -> None:
    if video is None:
        return
    for name in _DEPTH_VIDEO_TENSORS:
        _drop_attr(video, name)


def clear_droid_owner(owner: Any) -> None:
    """Drop one DROID frontend/backend/filter's graph, video, and cached features."""
    if owner is None:
        return
    clear_factor_graph(getattr(owner, "graph", None))
    for name in _OWNER_CACHED_TENSORS:
        _drop_attr(owner, name)
    _drop_attr(owner, "graph")
    _drop_attr(owner, "video")


def release_droid(droid: Any) -> None:
    """Drop DROID graphs, video buffers, cached features, and CUDA blocks."""
    import gc

    import torch

    if droid is None:
        return
    video = getattr(droid, "video", None)
    for name in ("frontend", "backend", "filterx", "traj_filler"):
        clear_droid_owner(getattr(droid, name, None))
        _drop_attr(droid, name)
    net = getattr(droid, "net", None)
    to_cpu = getattr(net, "cpu", None)
    if callable(to_cpu):
        try:
            to_cpu()
        except Exception:
            pass
    _drop_attr(droid, "net")
    _clear_depth_video(video)
    _drop_attr(droid, "video")
    if video is not None:
        destructor = getattr(video, "__del__", None)
        if callable(destructor):
            try:
                destructor()
            except Exception:
                pass
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
