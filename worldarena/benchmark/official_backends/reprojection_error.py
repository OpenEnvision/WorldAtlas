from typing import List
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from droid import Droid

from worldarena.benchmark.official_backends.base import (
    BaseMetric,
    official_checkpoint_path,
)


def image_stream(image_list, stride, calib):
    """image generator"""

    K = np.eye(3)

    image_list = image_list[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)

        h0, w0, _ = image.shape
        if calib is None:
            fx, fy, cx, cy = float(w0), float(h0), float(w0) / 2.0, float(h0) / 2.0
        else:
            fx, fy, cx, cy = calib
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        h1 = int(h0 * np.sqrt((512 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((512 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        yield t, image[None], intrinsics


def _valid_reprojection_errors(graph):
    """Return dense BA residuals for valid, confidence-weighted observations.

    Upstream DROID-SLAM's ``Droid.terminate`` returns only a trajectory. An
    earlier adapter expected a non-upstream ``(trajectory, valid_errors)``
    return shape, which made standard DROID-SLAM evaluations fail after
    tracking successfully. The factor graph retains learned correspondence
    targets and its final reprojection, so calculate the residuals before it
    is released.
    """
    coords, visible = graph.video.reproject(graph.ii, graph.jj)
    residuals = torch.linalg.vector_norm(coords - graph.target, dim=-1)
    confidence = graph.weight.mean(dim=-1)
    valid = (visible.squeeze(-1) > 0.5) & (confidence > 0) & torch.isfinite(residuals)
    return residuals[valid]


def _bounded_neighborhood_edges(
    frame_count: int,
    radius: int,
    budget: int,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evenly retain bidirectional local BA edges within a hard budget."""
    undirected_pairs: list[tuple[int, int]] = []
    for source in range(frame_count):
        for offset in range(1, radius + 1):
            target = source - offset
            if target >= 0:
                undirected_pairs.append((source, target))

    if not undirected_pairs:
        raise RuntimeError("DROID-SLAM needs at least two retained keyframes for reprojection")
    edge_pairs = max(1, budget // 2)
    if len(undirected_pairs) > edge_pairs:
        indices = torch.linspace(0, len(undirected_pairs) - 1, steps=edge_pairs).round().long().tolist()
        undirected_pairs = [undirected_pairs[index] for index in indices]
    pairs = [edge for source, target in undirected_pairs for edge in ((source, target), (target, source))]
    edges = torch.as_tensor(pairs, dtype=torch.long, device=device)
    return edges[:, 0], edges[:, 1]


def _run_bounded_global_ba(droid, args, *, steps: int, collect_errors: bool) -> torch.Tensor | None:
    """Run DROID's global BA with a bounded factor graph.

    ``DroidBackend`` materializes all proximity factors before observing its
    ``max_factors`` limit. On a few high-motion clips that graph exhausts an
    80 GiB A100. Retaining evenly distributed local bidirectional factors
    preserves trajectory-wide dense BA while bounding allocation up front.
    """
    from factor_graph import FactorGraph

    video = droid.video
    if not video.stereo and not torch.any(video.disps_sens):
        video.normalize()
    graph = FactorGraph(
        video,
        droid.net.update,
        corr_impl="alt",
        max_factors=min(16 * video.counter.value, max(1, int(args.max_factors))),
        upsample=args.upsample,
    )
    try:
        ii, jj = _bounded_neighborhood_edges(
            video.counter.value,
            radius=max(1, int(args.backend_radius)),
            budget=max(1, int(args.max_factors)),
            device=graph.device,
        )
        graph.add_factors(ii, jj)
        graph.update_lowmem(steps=steps)
        errors = _valid_reprojection_errors(graph) if collect_errors else None
        video.dirty[: video.counter.value] = True
        return errors
    finally:
        graph.clear_edges()


class ReprojectionErrorMetric(BaseMetric):
    """

    return: Reprojection error

    """

    def __init__(self) -> None:
        super().__init__()
        args = {
            "t0": 0,
            # Track a fixed, evenly spaced temporal subset. DROID's frontend
            # accumulates a large correlation volume before the bounded global
            # BA below runs; tracking every frame of a 121-frame high-motion
            # clip can therefore exhaust an 80 GiB GPU. A stride of four
            # applies the same ~30-frame budget to every prediction while
            # retaining video-wide temporal coverage.
            "stride": 4,
            "weights": official_checkpoint_path("droid.pth"),
            "buffer": 512,
            "beta": 0.3,
            "filter_thresh": 0.01,
            "warmup": 8,
            "keyframe_thresh": 4.0,
            "frontend_thresh": 16.0,
            "frontend_window": 25,
            "frontend_radius": 2,
            "frontend_nms": 1,
            "backend_thresh": 22.0,
            "backend_radius": 2,
            "backend_nms": 3,
            # Dense BA can otherwise retain roughly 16 factors per keyframe.
            # Long, high-motion clips then exceed an 80 GiB GPU despite the
            # fixed 512-pixel DROID input resolution. This cap keeps the
            # globally distributed factors representative and bounded.
            "max_factors": 512,
            # need high resolution depths
            "upsample": True,
            "stereo": False,
            "calib": None,
        }
        args = argparse.Namespace(**args)

        self._args = args
        self.droid = None
        try:
            torch.multiprocessing.set_start_method("spawn")
        except Exception as e:
            print(f"Error setting start method: {e}")

    def _compute_scores(
        self,
        rendered_images: List[str],
    ) -> float:

        for t, image, intrinsics in tqdm(
            image_stream(rendered_images, self._args.stride, self._args.calib)
        ):
            if t < self._args.t0:
                continue

            if self.droid is None:
                self._args.image_size = [image.shape[2], image.shape[3]]
                self.droid = Droid(self._args)
            self.droid.track(t, image, intrinsics=intrinsics)

        # The pinned upstream ``Droid.terminate`` returns only a trajectory,
        # not residuals. Reproduce its two global BA passes with bounded
        # graphs, exposing the valid dense residuals from the final pass.
        del self.droid.frontend
        torch.cuda.empty_cache()
        _run_bounded_global_ba(self.droid, self._args, steps=7, collect_errors=False)
        torch.cuda.empty_cache()
        try:
            valid_errors = _run_bounded_global_ba(
                self.droid,
                self._args,
                steps=12,
                collect_errors=True,
            )
            assert valid_errors is not None
            if valid_errors.numel() == 0:
                raise RuntimeError("DROID-SLAM produced no valid reprojection residuals")
            mean_error = valid_errors.mean().item()
        finally:
            self.droid = None
            torch.cuda.empty_cache()

        return mean_error
