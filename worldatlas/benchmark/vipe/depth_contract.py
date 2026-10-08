"""Official NVIDIA ViPE depth-model surface that Arena wrappers must expose.

``SLAMSystem`` reads ``supported_camera_types`` / ``depth_type`` before
``estimate``. ``AdaptiveDepthProcessor`` reads ``estimate(...).metric_depth``
or ``relative_inv_depth``.
"""

from __future__ import annotations

VIPE_DEPTH_MODEL_ATTRS = ("supported_camera_types", "depth_type", "estimate")


def pinhole_supported_camera_types():
    from vipe.utils.cameras import CameraType

    return [CameraType.PINHOLE]


def official_vipe_depth_model_public_attrs() -> tuple[str, ...]:
    """Public DepthEstimationModel names minus ABC helpers."""
    from vipe.priors.depth.base import DepthEstimationModel

    skip = {"register"}
    return tuple(
        name
        for name in dir(DepthEstimationModel)
        if not name.startswith("_") and name not in skip
    )
