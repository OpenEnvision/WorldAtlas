import numpy as np
import torch

from worldatlas.benchmark.official_backends.base import BaseMetric

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import structural_similarity_index_measure


class StructuralSimilarityIndexMeasureMetric(BaseMetric):
    """
    The Structural Similarity Index Measure (SSIM) is a method for predicting the
    perceived quality of digital television and cinematic pictures, as well as other
    kinds of digital images and videos. It is also used for measuring the similarity
    between two images. The SSIM index is a full reference metric; in other words, the
    measurement or prediction of image quality is based on an initial uncompressed or
    distortion-free image as reference.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = StructuralSimilarityIndexMeasure().to(self._device)

    def _compute_scores(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        img1, img2 = self._process_np_to_tensor(rendered_image, reference_image)
        if not bool(torch.isfinite(img1).all() and torch.isfinite(img2).all()):
            raise ValueError("SSIM inputs must be finite")
        # TorchMetrics infers max(range(image1), range(image2)). Two constant
        # frames have range zero, so its stabilizers vanish and black/black is
        # 0/0. Use the known ToTensor uint8 range only for this degenerate case;
        # retain the historical calculation for every nonconstant pair.
        constant_pair = bool(img1.max() == img1.min() and img2.max() == img2.min())
        if constant_pair:
            if rendered_image.dtype != np.uint8 or reference_image.dtype != np.uint8:
                raise ValueError("Constant SSIM inputs require uint8 images with a known data range")
            score = structural_similarity_index_measure(img1, img2, data_range=1.0).detach().item()
        else:
            score = self._metric(img1, img2).detach().item()
        if not np.isfinite(score):
            raise ValueError("SSIM produced a non-finite score")
        return score
