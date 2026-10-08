"""Per-frame metric-depth fallback for memory ViPE, without editing NVIDIA ViPE."""

from dataclasses import replace
import json
import logging
import sys
from contextvars import ContextVar

import torch

FALLBACK_EVENTS: ContextVar[list | None] = ContextVar("vipe_fallback_events", default=None)
_LOG = logging.getLogger(__name__)


class SparseDepthFallback:
    """Reuse the adaptive processor's resident metric model for sparse frames."""

    def __init__(self, prior, metric, intrinsics):
        self.prior = prior
        self.metric = metric
        self.intrinsics = intrinsics
        self.call_index = 0

    def estimate(self, src):
        self.call_index += 1
        depth = src.prompt_metric_depth
        threshold = self.prior.model.sampler.min_depth
        required = self.prior.model.args.K
        valid = torch.isfinite(depth) & (depth > threshold)
        count = int(valid.sum().item())
        clean = replace(src, prompt_metric_depth=torch.where(valid, depth, 0.0))
        reason = "insufficient_sparse_depth"
        if count >= required:
            try:
                return self.prior.estimate(clean)
            except ValueError as exc:
                if str(exc) != "There are not enough known points.":
                    raise
                reason = "insufficient_sparse_depth_after_sampling"
        result = self.metric.estimate(replace(
            src, prompt_metric_depth=None, intrinsics=self.intrinsics,
        ))
        value = result.metric_depth
        if value is None or value.shape != depth.shape or not bool(
            (torch.isfinite(value) & (value > 0)).all()
        ):
            raise ValueError("UniDepth sparse-depth fallback returned invalid metric depth")
        event = dict(prior_call_index=self.call_index, valid_points=count,
                     required_points=int(required), reason=reason, fallback="unidepth")
        events = FALLBACK_EVENTS.get()
        if events is not None:
            events.append(event)
        _LOG.warning("memory_vipe_depth_fallback %s", json.dumps(event))
        return result


def install_sparse_depth_fallback():
    """Bind the memory-only processor; retain all upstream alignment operations."""
    from vipe.pipeline import processors as processors_mod

    original = processors_mod.AdaptiveDepthProcessor
    if getattr(original, "_arena_sparse_fallback", False):
        return

    class MemoryAdaptiveDepthProcessor(original):
        _arena_sparse_fallback = True

        def __init__(self, slam_output, view_idx=0, *args, **kwargs):
            super().__init__(slam_output, view_idx, *args, **kwargs)
            prior = self.prompt_model
            if isinstance(prior, SparseDepthFallback):
                prior = prior.prior
            self.prompt_model = SparseDepthFallback(
                prior, self.depth_model, slam_output.intrinsics[view_idx],
            )

    processors_mod.AdaptiveDepthProcessor = MemoryAdaptiveDepthProcessor
    default_mod = sys.modules.get("vipe.pipeline.default")
    if default_mod is not None and hasattr(default_mod, "AdaptiveDepthProcessor"):
        default_mod.AdaptiveDepthProcessor = MemoryAdaptiveDepthProcessor
