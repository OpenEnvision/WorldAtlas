from typing import List
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from raft import RAFT
from utils.utils import load_ckpt

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.benchmark.official_backends.base import BaseMetric
from worldatlas.benchmark.official_backends.sea_raft_compat import (
    install_sea_raft_compat,
    load_sea_raft_eval_args,
)
from worldatlas.common.progress import log_progress


install_sea_raft_compat()


class OpticalFlowMetric(BaseMetric):
    """

    Using the median of estimated optical-flow to measure the motion magnitude.

    Optical-flow estimation -- SEA-RAFT

    RANGE: [0, ~] higher the better
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            args = load_sea_raft_eval_args()
            log_progress(
                "metric_load",
                metric="motion_magnitude",
                status="from_ckpt",
                checkpoint=str(args.path),
                device=self._device,
                note="torch.load is CPU; CUDA alloc starts at model.to()",
            )
            model = RAFT(args)
            load_ckpt(model, args.path)
            model.to(self._device)
            model.eval()
            self._model = model
            self._args = args
        except MetricLoadError:
            raise
        except Exception as exc:
            raise MetricLoadError(
                f"SEA-RAFT failed to load: {type(exc).__name__}: {exc}"
            ) from exc

    def load_image(self, imfile):
        image = cv2.imread(imfile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = image[None].to(self._device)
        return image

    def forward_flow(self, image1, image2):
        with torch.amp.autocast(device_type="cuda"):
            output = self._model(image1, image2, iters=self._args.iters, test_mode=True)
        flow_final = output["flow"][-1]
        info_final = output["info"][-1]
        return flow_final, info_final

    def _compute_flow(self, image1, image2):
        print("computing flow...")

        img1 = F.interpolate(
            image1,
            scale_factor=2**self._args.scale,
            mode="bilinear",
            align_corners=False,
        )
        img2 = F.interpolate(
            image2,
            scale_factor=2**self._args.scale,
            mode="bilinear",
            align_corners=False,
        )
        H, W = img1.shape[2:]
        flow, _ = self.forward_flow(img1, img2)
        flow_down = F.interpolate(
            flow,
            scale_factor=0.5**self._args.scale,
            mode="bilinear",
            align_corners=False,
        ) * (0.5**self._args.scale)

        flow = flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)
        return flow

    def _compute_scores(
        self,
        rendered_images: List[str],
        *,
        flow_pixel_scale: tuple[float, float] = (1.0, 1.0),
    ) -> float:

        if len(rendered_images) < 2:
            raise ValueError("motion_magnitude requires at least two frames")
        pixel_scale = np.asarray(flow_pixel_scale, dtype=np.float32)
        if pixel_scale.shape != (2,) or not np.all(np.isfinite(pixel_scale)) or np.any(pixel_scale <= 0):
            raise ValueError("flow_pixel_scale must contain two finite positive values")
        scores = []

        with torch.no_grad():
            images = rendered_images

            for i, (imfile1, imfile2) in tqdm(
                enumerate(zip(images[:-1], images[1:])),
                total=len(images) - 1,
                desc="Computing flow...",
            ):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)

                flow = self._compute_flow(image1, image2)
                # Restore each vector component before taking the norm/median;
                # rounded resize dimensions can have different x/y ratios.
                flow = flow * pixel_scale
                flow_magnitude = np.sqrt((flow[..., 0] ** 2 + flow[..., 1] ** 2))
                median_flow = float(torch.from_numpy(flow_magnitude).median().item())
                scores.append(median_flow)

        score = sum(scores) / len(scores)
        return score
