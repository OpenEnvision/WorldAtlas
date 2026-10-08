"""Release SAM2 per-video CUDA state after masks are copied off GPU.

Motion Accuracy then runs SEA-RAFT on the numpy masks. Leaving SAM2 images,
cached features, and tracking outputs alive next to a native-resolution flow
volume OOMs 80 GiB cards on long/high-res clips (LongCat 1/GPU, ~74 GiB).
Scoring is unchanged: masks are CPU numpy before this runs.
"""

from __future__ import annotations

from typing import Any


def release_sam2_inference_state(predictor: Any, inference_state: Any) -> None:
    """Drop per-video SAM2 tensors. Does not change how scores were computed."""
    if inference_state is None:
        return
    reset = getattr(predictor, "reset_state", None)
    if callable(reset):
        try:
            reset(inference_state)
        except Exception:
            pass
    if not isinstance(inference_state, dict):
        return
    for key in (
        "images",
        "cached_features",
        "constants",
        "output_dict",
        "output_dict_per_obj",
        "temp_output_dict_per_obj",
        "point_inputs_per_obj",
        "mask_inputs_per_obj",
    ):
        value = inference_state.get(key)
        if isinstance(value, dict):
            value.clear()
        inference_state[key] = None
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
