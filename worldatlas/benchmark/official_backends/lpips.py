import numpy as np

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.benchmark.official_backends.base import BaseMetric
from worldatlas.benchmark.local_checkpoints import ensure_local_alexnet_weights

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LearnedPerceptualImagePatchSimilarityMetric(BaseMetric):
    """
    The Learned Perceptual Image Patch Similarity (LPIPS) calculates perceptual
    similarity between two images. LPIPS essentially computes the similarity between
    the activations of two image patches for some pre-defined network. This measure has
    been shown to match human perception well. A low LPIPS score means that image
    patches are perceptual similar.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            ensure_local_alexnet_weights()
            self._metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
                self._device
            )
        except MetricLoadError:
            raise
        except Exception as exc:
            raise MetricLoadError(
                f"LPIPS failed to load: {type(exc).__name__}: {exc}"
            ) from exc

    def _compute_scores(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        img1, img2 = self._process_np_to_tensor(rendered_image, reference_image)

        score: float = self._metric(img1, img2).detach().item()
        return score
