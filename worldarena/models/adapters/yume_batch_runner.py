"""Batch subprocess runner for Yume inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from worldarena.common.checkpoints import apply_checkpoint_env
from worldarena.models.adapters.yume_runner import (
    _load_persistent_runtime,
    _run_persistent_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena YUME persistent batch runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--entrypoint", default="webapp_single_gpu.py", type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    return parser.parse_args()


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty YUME batch spec: {path}")
    return rows


def _generation(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("generation", {})
    if not isinstance(payload, dict):
        raise ValueError(f"YUME row has invalid generation payload: {row.get('sample_id')}")
    return payload


def _value(row: dict[str, Any], name: str, default: Any) -> Any:
    return _generation(row).get(name, default)


def _single_runtime_value(rows: list[dict[str, Any]], name: str, default: Any) -> Any:
    values = {_generation(row).get(name, default) for row in rows}
    if len(values) > 1:
        raise ValueError(f"YUME persistent batch cannot vary {name} within one worker: {sorted(values)!r}")
    return next(iter(values))


def main() -> None:
    args = parse_args()
    os.environ.update(apply_checkpoint_env())

    repo_root = Path(args.repo_root).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    rows = _load_batch_spec(batch_spec_path)

    gpu_index = int(_single_runtime_value(rows, "gpu_index", 0))
    load_caption_model = any(bool(_value(row, "load_caption_model", False)) for row in rows)
    refine_from_image = any(bool(_value(row, "refine_from_image", False)) for row in rows)
    caption_model_dir = next(
        (
            _generation(row).get("caption_model_dir")
            for row in rows
            if _generation(row).get("caption_model_dir") is not None
        ),
        None,
    )

    with tempfile.TemporaryDirectory(prefix="worldarena_yume_batch_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        runtime = _load_persistent_runtime(
            repo_root=repo_root,
            checkpoint_dir=args.checkpoint_dir,
            entrypoint=args.entrypoint,
            generated_root=temp_dir / "outputs",
            gpu_index=gpu_index,
            caption_model_dir=caption_model_dir,
            load_caption_model=load_caption_model,
            refine_from_image=refine_from_image,
        )

        for index, row in enumerate(rows, start=1):
            sample_id = str(row.get("sample_id", f"row-{index}"))
            output_path = Path(row["output_path"]).expanduser().resolve()
            conditioning_image = row.get("conditioning_image")
            conditioning_path = Path(conditioning_image).expanduser().resolve() if conditioning_image else None
            prompt = str(row.get("prompt", ""))
            try:
                payload = _run_persistent_generation(
                    runtime,
                    conditioning_image=conditioning_path,
                    output_path=output_path,
                    prompt=prompt,
                    mode=str(_value(row, "mode", "I2V")),
                    fps=int(_value(row, "fps", 16)),
                    sample_steps=int(_value(row, "sample_steps", 4)),
                    sample_num=int(_value(row, "sample_num", 1)),
                    frame_zero=int(_value(row, "frame_zero", 32)),
                    shift=float(_value(row, "shift", 5.0)),
                    seed=int(_value(row, "seed", -1)),
                    resolution=str(_value(row, "resolution", "704x1280")),
                    refine_from_image=bool(_value(row, "refine_from_image", False)),
                    memory_optimization=bool(_value(row, "memory_optimization", False)),
                    vae_memory_optimization=bool(_value(row, "vae_memory_optimization", False)),
                    camera_movement1=str(_value(row, "camera_movement1", "None")),
                    camera_movement2=str(_value(row, "camera_movement2", "·")),
                )
                print(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "generated",
                            "prediction_path": payload["prediction_path"],
                            "index": index,
                            "total": len(rows),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "failed",
                            "error": str(exc),
                            "index": index,
                            "total": len(rows),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
