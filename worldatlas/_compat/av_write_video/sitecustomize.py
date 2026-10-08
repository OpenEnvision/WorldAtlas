"""Patch torchvision.io.write_video for PyAV 14+.

torchvision 0.20 still does ``frame.pict_type = "NONE"``. PyAV 14+ requires an
integer / PictureType, which is what killed DreamX AR after the 60 s rollout
finished decoding.
"""

from __future__ import annotations

from typing import Any


def _write_video_with_int_pict_type(
    filename: str,
    video_array: Any,
    fps: float,
    video_codec: str = "libx264",
    options: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    import av
    import numpy as np

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in model envs
        torch = None

    if torch is not None and isinstance(video_array, torch.Tensor):
        frames = video_array.detach().cpu().numpy()
    else:
        frames = np.asarray(video_array)
    frames = np.ascontiguousarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"write_video expects [T, H, W, C], got shape {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    none_type: Any = 0
    try:
        from av.video.frame import PictureType

        none_type = PictureType.NONE
    except Exception:
        none_type = 0

    container = av.open(str(filename), mode="w")
    try:
        stream = container.add_stream(video_codec, rate=int(round(float(fps))))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        if options:
            stream.options = {str(key): str(value) for key, value in options.items()}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pict_type = none_type
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def install_torchvision_write_video_compat() -> None:
    try:
        import torchvision.io as io
        import torchvision.io.video as video_mod
    except Exception:
        return
    if getattr(video_mod.write_video, "_worldatlas_av17_patched", False):
        return

    original = video_mod.write_video

    def write_video(*args: Any, **kwargs: Any):
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            if "integer is required" not in str(exc):
                raise
            return _write_video_with_int_pict_type(*args, **kwargs)

    write_video._worldatlas_av17_patched = True  # type: ignore[attr-defined]
    video_mod.write_video = write_video
    io.write_video = write_video


install_torchvision_write_video_compat()
