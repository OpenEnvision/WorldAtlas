"""Batch subprocess that rolls out Open Dreamer for WorldArena samples.

Upstream ``next-state/open-dreamer`` ships a training pipeline plus the ``dreamer``
package (causal video tokenizer, action-conditioned latent dynamics, Orbax
checkpoint bundles) but no benchmark inference entry point, so this module is the
entry point: it loads the checkpoint once per batch and, for every sample,
encodes the conditioning frames, unrolls the dynamics under WorldArena-derived
VPT actions, decodes the predicted latents, and writes one mp4.

``jax``, the upstream ``dreamer`` package, and ``imageio`` are imported lazily so
that argument handling and the dry-run path stay usable from the core evaluation
environment, which has none of them installed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)
from worldarena.models.adapters.open_dreamer_actions import (
    NATIVE_ACTION_SPACE,
    build_vpt_action_sequence,
)


DRY_RUN_ENV = "WORLDARENA_OPEN_DREAMER_DRY_RUN"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
# Released Open Dreamer checkpoints are trained on VPT frames padded from
# 360x640 to 368x640.  Both values are re-read from the restored tokenizer and
# only used verbatim on the dry-run path, which never loads a checkpoint.
DEFAULT_MODEL_HEIGHT = 368
DEFAULT_MODEL_WIDTH = 640
DEFAULT_CONTENT_HEIGHT = 360


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    return parser.parse_args()


def _slug(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "sample"


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def dry_run_enabled() -> bool:
    return os.environ.get(DRY_RUN_ENV, "") == "1"


# ---------------------------------------------------------------------------
# Checkpoint layout
# ---------------------------------------------------------------------------


def resolve_checkpoint_layout(checkpoint_dir: Path, *, use_ema: bool) -> dict[str, Any]:
    """Locate the Orbax checkpoint root and verify the items the rollout needs.

    ``DynamicsCheckpointBundle.from_pretrained`` opens an ``ocp.CheckpointManager``
    on a directory of numbered steps and restores its latest one, so the path we
    hand it must be the parent of ``<step>/``.  Upstream ``eval_fvd.py`` also
    accepts a run directory holding ``checkpoints/``; the same fallback is applied
    here, and the requested items are validated up front so a missing
    ``dynamics``/``dynamics_ema`` fails before JAX allocates any device memory.
    """
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Open Dreamer checkpoint directory not found: {checkpoint_dir}")

    candidates = [checkpoint_dir, checkpoint_dir / "checkpoints"]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        steps = sorted(
            (int(path.name), path)
            for path in candidate.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        if steps:
            step, step_dir = steps[-1]
            dynamics_item = "dynamics_ema" if use_ema else "dynamics"
            missing = [
                item for item in (dynamics_item, "tokenizer") if not (step_dir / item).is_dir()
            ]
            if missing:
                available = sorted(path.name for path in step_dir.iterdir() if path.is_dir())
                raise FileNotFoundError(
                    f"Open Dreamer checkpoint {step_dir} is missing item(s) {missing}; "
                    f"available items: {available}"
                    + (
                        " (set generation.use_ema to match the item the checkpoint ships)"
                        if dynamics_item not in available and available
                        else ""
                    )
                )
            return {
                "checkpoint_path": candidate,
                "step": step,
                "step_dir": step_dir,
                "dynamics_item": dynamics_item,
            }

    raise FileNotFoundError(
        "no Orbax step directory found under "
        + " or ".join(str(path) for path in candidates)
        + "; expected a numbered checkpoint directory such as '250000/'"
    )


# ---------------------------------------------------------------------------
# Conditioning frames
# ---------------------------------------------------------------------------


def _read_conditioning_frames(path: Path, *, count: int) -> tuple[list[Any], float | None]:
    """Read up to ``count`` RGB frames plus the source fps, if available."""
    from PIL import Image  # noqa: PLC0415 - pillow ships with imageio

    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            return [image.convert("RGB").copy()], None

    import imageio.v3 as iio  # noqa: PLC0415 - upstream dependency only

    try:
        metadata = iio.immeta(path)
        fps = float(metadata.get("fps") or 0.0) or None
    except Exception:
        fps = None

    frames: list[Any] = []
    for index, frame in enumerate(iio.imiter(path)):
        if index >= count:
            break
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(
                f"expected RGB frames with shape HxWx3 in {path}, got {array.shape} at frame {index}"
            )
        frames.append(Image.fromarray(array[..., :3].astype(np.uint8)))
    if not frames:
        raise ValueError(f"no readable RGB frames in conditioning asset: {path}")
    return frames, fps


def _fit_frame(frame: Any, *, width: int, height: int, fit_mode: str) -> Any:
    """Resize one frame to ``width x height`` under the configured fit mode."""
    from PIL import Image  # noqa: PLC0415

    source_width, source_height = frame.size
    if (source_width, source_height) == (width, height):
        return frame

    if fit_mode == "crop":
        target_ratio = width / height
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = int(round(source_height * target_ratio))
            offset = (source_width - crop_width) // 2
            box = (offset, 0, offset + crop_width, source_height)
        else:
            crop_height = int(round(source_width / target_ratio))
            offset = (source_height - crop_height) // 2
            box = (0, offset, source_width, offset + crop_height)
        return frame.resize((width, height), Image.LANCZOS, box=box)

    if fit_mode == "pad":
        scale = min(width / source_width, height / source_height)
        inner_width = max(1, int(round(source_width * scale)))
        inner_height = max(1, int(round(source_height * scale)))
        resized = frame.resize((inner_width, inner_height), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(resized, ((width - inner_width) // 2, (height - inner_height) // 2))
        return canvas

    raise ValueError(f"unsupported fit_mode {fit_mode!r}; expected 'crop' or 'pad'")


def prepare_context_frames(
    path: Path,
    *,
    context_frames: int,
    model_height: int,
    model_width: int,
    content_height: int,
    fit_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the ``(T, H, W, 3)`` uint8 context clip the tokenizer expects.

    Frames are fitted to the unpadded VPT content size and then zero-padded to the
    trained frame height, matching the ``padding_H`` transform used for training.
    A single conditioning image is repeated, which is what the harness itself does
    when it freezes an image asset into a conditioning video.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"conditioning asset not found: {path}")
    if content_height > model_height:
        raise ValueError(
            f"content_height {content_height} exceeds model height {model_height}"
        )

    frames, source_fps = _read_conditioning_frames(path, count=context_frames)
    source_width, source_height = frames[0].size
    repeated = len(frames) < context_frames
    while len(frames) < context_frames:
        frames.append(frames[-1])

    fitted = [
        np.asarray(
            _fit_frame(frame, width=model_width, height=content_height, fit_mode=fit_mode),
            dtype=np.uint8,
        )
        for frame in frames[:context_frames]
    ]
    clip = np.stack(fitted, axis=0)

    pad_total = model_height - content_height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    if pad_total:
        clip = np.pad(
            clip,
            ((0, 0), (pad_top, pad_bottom), (0, 0), (0, 0)),
            mode="constant",
        )

    details = {
        "conditioning_path": str(path),
        "conditioning_frames_read": len(fitted),
        "conditioning_frames_repeated": bool(repeated),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "source_fps": source_fps,
        "fit_mode": fit_mode,
        "content_height": int(content_height),
        "model_height": int(model_height),
        "model_width": int(model_width),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
    }
    return clip, details


# ---------------------------------------------------------------------------
# Upstream runtime
# ---------------------------------------------------------------------------


class OpenDreamerRuntime:
    """Resident tokenizer + dynamics pair shared by every sample in one batch."""

    def __init__(
        self,
        *,
        repo_root: Path,
        checkpoint_layout: dict[str, Any],
        generation: dict[str, Any],
    ) -> None:
        self.repo_root = repo_root
        self.generation = generation
        self._enter_repo_on_path()

        import jax  # noqa: PLC0415 - upstream dependency only

        from dreamer.checkpointing import DynamicsCheckpointBundle  # noqa: PLC0415
        from dreamer.generation import DenoiseSchedule  # noqa: PLC0415
        from dreamer.parallel import build_parallel  # noqa: PLC0415
        from dreamer.sampler import decode_jit, encode_jit  # noqa: PLC0415

        self._jax = jax
        self._encode_jit = encode_jit
        self._decode_jit = decode_jit

        if not _as_bool(generation.get("allow_cpu"), default=False):
            self._require_gpu()

        strategy = str(generation.get("parallel_strategy", "data"))
        mesh, _, mesh_rules = build_parallel(strategy)
        self._mesh = mesh
        self._mesh_context = self._build_mesh_context(mesh)
        self._mesh_context.__enter__()

        dynamics_item = checkpoint_layout["dynamics_item"]
        bundle = DynamicsCheckpointBundle.from_pretrained(
            str(checkpoint_layout["checkpoint_path"]),
            mesh_rules=mesh_rules,
            model_names={"tokenizer", dynamics_item},
        )
        self.tokenizer = bundle.tokenizer
        self.dynamics = getattr(bundle, dynamics_item)
        if self.tokenizer is None or self.dynamics is None:
            raise RuntimeError(
                f"Open Dreamer checkpoint did not restore tokenizer and {dynamics_item}"
            )

        num_steps = int(generation.get("num_steps", 4))
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.schedule = DenoiseSchedule.init(num_steps, self.dynamics.cfg.k_max)
        self.num_steps = num_steps
        self.dynamics_item = dynamics_item
        self.checkpoint_step = checkpoint_layout["step"]
        self.model_height = int(self.tokenizer.decoder.H)
        self.model_width = int(self.tokenizer.decoder.W)
        # ``continuous_action_dim > 0`` is how upstream configs select the
        # continuous mouse parameterization over the mu-law categorical bins.
        self.mouse_repr = (
            "continuous" if int(self.dynamics.cfg.continuous_action_dim or 0) > 0 else "categorical"
        )

    def _enter_repo_on_path(self) -> None:
        repo_root = str(self.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _require_gpu(self) -> None:
        try:
            devices = self._jax.devices("gpu")
        except RuntimeError as exc:
            raise RuntimeError(
                "no JAX GPU devices are available; Open Dreamer rollouts need a CUDA-visible "
                "JAX build (set generation.allow_cpu to override)"
            ) from exc
        if not devices:
            raise RuntimeError(
                "no JAX GPU devices are available; Open Dreamer rollouts need a CUDA-visible "
                "JAX build (set generation.allow_cpu to override)"
            )

    def _build_mesh_context(self, mesh: Any) -> Any:
        # ``jax.set_mesh`` replaced using the mesh object directly as a context
        # manager; upstream scripts assume the newer API, so fall back for older
        # JAX builds rather than failing at import time.
        if hasattr(self._jax, "set_mesh"):
            return self._jax.set_mesh(mesh)
        return mesh

    def close(self) -> None:
        context = getattr(self, "_mesh_context", None)
        if context is not None:
            context.__exit__(None, None, None)
            self._mesh_context = None

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint_step": int(self.checkpoint_step),
            "dynamics_item": self.dynamics_item,
            "num_steps": self.num_steps,
            "k_max": int(self.dynamics.cfg.k_max),
            "dynamics_context_length": int(self.dynamics.cfg.context_length),
            "mouse_repr": self.mouse_repr,
            "model_height": self.model_height,
            "model_width": self.model_width,
        }

    def _batched_actions(self, action_dicts: list[dict[str, Any]]) -> Any:
        import jax.numpy as jnp  # noqa: PLC0415

        from dreamer.actions import parse_action_dicts, shift_actions  # noqa: PLC0415

        actions = parse_action_dicts(action_dicts, self.mouse_repr)
        actions = self._jax.tree.map(
            lambda leaf: jnp.asarray(leaf)[None] if leaf is not None else None,
            actions,
        )
        # Shift before splitting so context and future actions keep the
        # training/eval alignment: entry t becomes the action that produced frame t.
        return shift_actions(actions, self.dynamics.cfg.categorical_action_dim)

    def _decode_generated(
        self,
        latents: Any,
        *,
        context_frames: int,
        horizon: int,
        chunk_size: int,
    ) -> np.ndarray:
        """Decode only the generated latents, in windows the decoder can cache."""
        import jax.numpy as jnp  # noqa: PLC0415

        if chunk_size < 1:
            raise ValueError(f"decode_chunk_size must be >= 1, got {chunk_size}")
        decoder_context = self.tokenizer.decoder.context_length
        overlap = max(0, int(decoder_context) - 1) if decoder_context is not None else 0

        chunks: list[np.ndarray] = []
        for offset in range(0, horizon, chunk_size):
            start = context_frames + offset
            end = context_frames + min(offset + chunk_size, horizon)
            window_start = max(0, start - overlap)
            decoded = self._decode_jit(self.tokenizer, latents[:, window_start:end])
            take_start = start - window_start
            generated = decoded[:, take_start : take_start + (end - start)]
            chunks.append(
                np.asarray(jnp.clip(generated[0], 0, 255).astype(jnp.uint8), copy=True)
            )
        return np.concatenate(chunks, axis=0)

    def rollout(
        self,
        frames: np.ndarray,
        action_dicts: list[dict[str, Any]],
        *,
        horizon: int,
        seed: int,
        decode_chunk_size: int,
        use_kv_cache: bool,
    ) -> np.ndarray:
        """Return the ``(horizon, H, W, 3)`` uint8 frames generated after ``frames``."""
        import jax.numpy as jnp  # noqa: PLC0415

        from dreamer.generation import latent_rollout  # noqa: PLC0415

        context_frames = int(frames.shape[0])
        expected_actions = context_frames + horizon
        if len(action_dicts) != expected_actions:
            raise ValueError(
                f"expected {expected_actions} VPT actions for "
                f"context={context_frames} horizon={horizon}, got {len(action_dicts)}"
            )
        if frames.shape[1:] != (self.model_height, self.model_width, 3):
            raise ValueError(
                f"context clip shape {frames.shape[1:]} does not match the trained frame "
                f"shape {(self.model_height, self.model_width, 3)}"
            )

        actions = self._batched_actions(action_dicts)
        latents_ctx = self._encode_jit(self.tokenizer, jnp.asarray(frames[None]))
        result = latent_rollout(
            self.dynamics,
            actions_future=actions[:, context_frames:expected_actions],
            schedule=self.schedule,
            latents_ctx=latents_ctx,
            actions_ctx=actions[:, :context_frames],
            num_steps=horizon,
            rng=self._jax.random.PRNGKey(int(seed)),
            use_kv_cache=use_kv_cache,
        )
        return self._decode_generated(
            result["latents"],
            context_frames=context_frames,
            horizon=horizon,
            chunk_size=decode_chunk_size,
        )


def write_video(path: Path, frames: np.ndarray, *, fps: float) -> None:
    height, width = int(frames.shape[1]), int(frames.shape[2])
    # ``macro_block_size`` makes the ffmpeg writer silently rescale frames whose
    # size is not a multiple of it, which would stretch a 360-row prediction back
    # to the 368-row padded geometry.  Disable that and require the even
    # dimensions libx264 needs for yuv420p instead.
    if height % 2 or width % 2:
        raise ValueError(
            f"prediction frame size {width}x{height} must have even dimensions for libx264"
        )

    import imageio.v3 as iio  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        path,
        frames.astype(np.uint8),
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
    )


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------


def _camera_path_for_row(row: dict[str, Any]) -> Any:
    return row.get("camera_path")


def plan_row(row: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    """Resolve the frame and action plan for one sample without loading weights."""
    context_frames = max(int(generation.get("context_frames", 4)), 1)
    horizon = max(int(generation.get("horizon", 64)), 1)
    actions, action_metadata = build_vpt_action_sequence(
        _camera_path_for_row(row),
        context_frames=context_frames,
        horizon=horizon,
        mouse_pan_pixels=float(generation.get("mouse_pan_pixels", 20.0)),
        mouse_tilt_pixels=float(generation.get("mouse_tilt_pixels", 13.0)),
    )
    action_metadata["camera_source"] = str(
        row.get("camera_source") or "worldarena_camera_path"
    )
    return {
        "context_frames": context_frames,
        "horizon": horizon,
        "actions": actions,
        "action_metadata": action_metadata,
    }


def write_action_files(
    output_path: Path,
    action_dir: Path,
    sample_id: str,
    plan: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Write the VPT action stream plus the WorldArena action sidecar."""
    action_dir.mkdir(parents=True, exist_ok=True)
    actions_path = action_dir / f"{_slug(sample_id)}.vpt.jsonl"
    with actions_path.open("w", encoding="utf-8") as handle:
        for action in plan["actions"]:
            handle.write(json.dumps(action, ensure_ascii=False) + "\n")

    payload = {**plan["action_metadata"], **extra, "vpt_actions_path": str(actions_path)}
    payload["actions"] = payload["camera_path"]
    sidecar = output_path.with_name(f"{output_path.stem}_actions.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"action_sidecar": str(sidecar), "vpt_actions_path": str(actions_path)}


def _run_row(
    row: dict[str, Any],
    *,
    runtime: OpenDreamerRuntime | None,
    generation: dict[str, Any],
    batch_spec_path: Path,
    dry_run: bool,
) -> None:
    sample_id = str(row["sample_id"])
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    plan = plan_row(row, generation)
    action_dir = batch_spec_path.parent / "open_dreamer_actions"
    fit_mode = str(generation.get("fit_mode", "crop"))
    crop_padding = _as_bool(generation.get("crop_model_padding"), default=True)
    fps = float(generation.get("fps", 20.0))

    if dry_run:
        geometry = {
            "model_height": int(generation.get("model_height", DEFAULT_MODEL_HEIGHT)),
            "model_width": int(generation.get("model_width", DEFAULT_MODEL_WIDTH)),
            "content_height": int(generation.get("content_height", DEFAULT_CONTENT_HEIGHT)),
            "fit_mode": fit_mode,
            "crop_model_padding": crop_padding,
            "output_fps": fps,
            "native_action_space": NATIVE_ACTION_SPACE,
        }
        written = write_action_files(output_path, action_dir, sample_id, plan, geometry)
        print_status(
            sample_id,
            "dry_run",
            context_frames=plan["context_frames"],
            horizon=plan["horizon"],
            total_frames=plan["action_metadata"]["total_frames"],
            camera_path=plan["action_metadata"]["camera_path"],
            **geometry,
            **written,
        )
        return

    assert runtime is not None
    content_height = int(generation.get("content_height", DEFAULT_CONTENT_HEIGHT))
    frames, frame_details = prepare_context_frames(
        Path(str(row["conditioning_image"])),
        context_frames=plan["context_frames"],
        model_height=runtime.model_height,
        model_width=runtime.model_width,
        content_height=content_height,
        fit_mode=fit_mode,
    )
    generated = runtime.rollout(
        frames,
        plan["actions"],
        horizon=plan["horizon"],
        seed=int(generation.get("seed", 0)),
        decode_chunk_size=int(generation.get("decode_chunk_size", 16)),
        use_kv_cache=_as_bool(generation.get("use_kv_cache"), default=True),
    )

    clip = np.concatenate([frames, generated], axis=0)
    if crop_padding and frame_details["pad_top"] + frame_details["pad_bottom"]:
        top = frame_details["pad_top"]
        clip = clip[:, top : top + content_height]

    run_dir = batch_spec_path.parent / "open_dreamer_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = run_dir / f"{_slug(sample_id)}.mp4"
    write_video(staged, clip, fps=fps)
    copy_output(staged, output_path)
    staged.unlink(missing_ok=True)

    details = {
        **frame_details,
        **runtime.describe(),
        "crop_model_padding": crop_padding,
        "output_fps": fps,
        "generated_frames": int(generated.shape[0]),
        "total_frames": int(clip.shape[0]),
        "output_height": int(clip.shape[1]),
        "output_width": int(clip.shape[2]),
    }
    written = write_action_files(output_path, action_dir, sample_id, plan, details)
    print_status(
        sample_id,
        "generated",
        prediction_path=str(output_path),
        camera_path=plan["action_metadata"]["camera_path"],
        **details,
        **written,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    dry_run = dry_run_enabled()

    if not (repo_root / "dreamer").is_dir():
        raise FileNotFoundError(
            f"Open Dreamer 'dreamer' package not found under {repo_root}; "
            "run 'git submodule update --init thirdparty/open-dreamer'"
        )

    runtime: OpenDreamerRuntime | None = None
    if not dry_run:
        if args.checkpoint_dir is None:
            raise ValueError("Open Dreamer requires --checkpoint_dir")
        checkpoint_layout = resolve_checkpoint_layout(
            Path(args.checkpoint_dir),
            use_ema=_as_bool(generation.get("use_ema"), default=True),
        )
        runtime = OpenDreamerRuntime(
            repo_root=repo_root,
            checkpoint_layout=checkpoint_layout,
            generation=generation,
        )

    rows = load_batch_spec(batch_spec_path)
    failures: list[tuple[str, str]] = []
    try:
        for row in rows:
            try:
                _run_row(
                    row,
                    runtime=runtime,
                    generation=generation,
                    batch_spec_path=batch_spec_path,
                    dry_run=dry_run,
                )
            except Exception as exc:
                sample_id = str(row.get("sample_id") or "unknown")
                failures.append((sample_id, str(exc)))
                print_status(sample_id, "failed", error=str(exc))
    finally:
        if runtime is not None:
            runtime.close()

    if failures:
        summary = "; ".join(f"{sample_id}: {error}" for sample_id, error in failures)
        raise RuntimeError(f"Open Dreamer failed for {len(failures)} sample(s): {summary}")


if __name__ == "__main__":
    main()
