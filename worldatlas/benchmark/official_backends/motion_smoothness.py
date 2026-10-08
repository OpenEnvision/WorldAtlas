from typing import List
import torch
import cv2
import numpy as np

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.benchmark.official_backends.base import (
    BaseMetric,
    mean_squared_pixel_error,
    official_checkpoint_path,
)

import config as cfg
from Trainer_finetune import Model, convert
from benchmark.utils.padder import InputPadder

from worldatlas.benchmark.official_backends.lpips import (
    LearnedPerceptualImagePatchSimilarityMetric,
)
from worldatlas.benchmark.official_backends.ssim import (
    StructuralSimilarityIndexMeasureMetric,
)

"""==========Model setting=========="""
TTA = True
cfg.MODEL_CONFIG["LOGNAME"] = "VFIMamba"
cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])


class MotionSmoothnessMetric(BaseMetric):
    """
    add description here
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            model = Model(-1)
            checkpoint = torch.load(
                official_checkpoint_path("VFIMamba.pkl"), map_location="cpu"
            )
            model.net.load_state_dict(convert(checkpoint), strict=True)
            model.eval()
            model.device()
            self._model = model
            self._ssim_metric = StructuralSimilarityIndexMeasureMetric()
            self._lpips_metric = LearnedPerceptualImagePatchSimilarityMetric()
        except MetricLoadError:
            raise
        except Exception as exc:
            raise MetricLoadError(
                f"VFIMamba failed to load: {type(exc).__name__}: {exc}"
            ) from exc

    def _compute_scores(
        self,
        rendered_images: List[str],
    ) -> float:
        scores_ssim = []
        scores_lpips = []
        scores_mse = []
        even_images = rendered_images[::2]
        odd_images = rendered_images[1::2]

        print("=========================Start Interpolating=========================")
        for i, (image1, image2) in enumerate(zip(even_images[:-1], even_images[1:])):
            odd_image = odd_images[i]
            mid_pred = cv2.imread(odd_image)
            I0 = cv2.imread(image1)
            I2 = cv2.imread(image2)

            I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)
            I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)

            padder = InputPadder(I0_.shape, divisor=32)
            I0_, I2_ = padder.pad(I0_, I2_)

            mid = (
                padder.unpad(
                    self._model.inference(
                        I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=0.0
                    )
                )[0]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 255.0
            )
            if not np.isfinite(mid).all():
                raise ValueError(f"VFIMamba produced non-finite pixels at midpoint {i}: {odd_image}")
            mid = mid.astype(np.uint8)

            ssim_score = self._ssim_metric._compute_scores(mid, mid_pred)
            lpips_score = self._lpips_metric._compute_scores(mid, mid_pred)
            # psnr_score = self._psnr_metric._compute_scores(mid, mid_pred)
            mse_score = mean_squared_pixel_error(mid, mid_pred)
            for name, value in (("ssim", ssim_score), ("lpips", lpips_score), ("mse", mse_score)):
                if not np.isfinite(value):
                    raise ValueError(f"Motion smoothness {name} is non-finite at midpoint {i}: {odd_image}")

            scores_ssim.append(ssim_score)
            scores_lpips.append(lpips_score)
            scores_mse.append(mse_score)
        print("=========================Done=========================")

        score_ssim = sum(scores_ssim) / len(scores_ssim)
        score_lpips = sum(scores_lpips) / len(scores_lpips)
        score_mse = sum(scores_mse) / len(scores_mse)
        return (score_mse, score_ssim, score_lpips)
