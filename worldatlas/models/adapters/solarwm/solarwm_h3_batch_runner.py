"""Bridge WorldAtlas to the official SolarWM-H3 source-length SP8 route."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from worldatlas.benchmark.annotations import load_camera_matrices
from worldatlas.models.adapters.batch_runner_common import (
    copy_output, load_batch_spec, load_json, print_status,
)
from worldatlas.models.adapters.solarwm.solarwm_batch_runner import (
    _override, _resolve_child,
    build_solarwm_environment, parse_args, resolve_runtime_paths,
    validate_solarwm_command,
)
from worldatlas.models.adapters.solarwm.solarwm_h3_supervisor import run_supervised


def build_input_bundle(rows, input_root: Path, generation: dict) -> tuple[Path, dict[str, Path]]:
    """Use the official H3 image/camera demo route without synthetic video frames."""
    if not rows:
        raise ValueError("SolarWM-H3 batch must not be empty")
    fps = float(generation.get("fps", 24))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("SolarWM-H3 FPS must be positive and finite")
    source_root = input_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    plan, outputs, seen = [], {}, set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate SolarWM-H3 sample_id: {sample_id!r}")
        seen.add(sample_id)
        frames = int(row["source_frame_count"])
        if frames < 22:
            raise ValueError("SolarWM-H3 requires at least 22 source frames")
        if frames > 960 and not generation.get("stream_decode", True):
            raise ValueError("SolarWM-H3 sources over 960 frames require stream_decode")
        poses = load_camera_matrices(row["annotation_path"], target_frames=frames)
        if poses is None or poses.shape != (frames, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(f"Invalid c2w trajectory for {sample_id}")
        np.testing.assert_allclose(
            poses[:, 3], np.broadcast_to([0, 0, 0, 1], (frames, 4)), atol=1e-5,
        )
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Empty prompt for {sample_id}")
        # The pending subset changes after partial publication. Keep source,
        # condition and rollout keys stable across retries of the same sample.
        key = f"sample-{hashlib.sha256(sample_id.encode()).hexdigest()[:24]}"
        image_name, camera_name, prompt_name = (
            f"{key}.image.png", f"{key}.camera.npy", f"{key}.prompt.txt",
        )
        with Image.open(Path(row["conditioning_image"]).resolve()) as source:
            source.convert("RGB").save(source_root / image_name)
        np.save(source_root / camera_name, poses.astype(np.float64), allow_pickle=False)
        (source_root / prompt_name).write_text(prompt, encoding="utf-8")
        item = dict(
            sample_id=sample_id, key=key, dataset="worldatlas", input_kind="demo",
            num_frames=frames, fps=fps, width=1344, height=768, caption=prompt,
            image=image_name, camera=camera_name, prompt=prompt_name,
            image_sha256=hashlib.sha256((source_root / image_name).read_bytes()).hexdigest(),
            camera_sha256=hashlib.sha256((source_root / camera_name).read_bytes()).hexdigest(),
            prompt_sha256=hashlib.sha256((source_root / prompt_name).read_bytes()).hexdigest(),
        )
        plan.append(item)
        outputs[sample_id] = Path("worldatlas") / f"{key}.mp4"
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[h3-input] prepared={index + 1}/{len(rows)}", flush=True)
    path = input_root / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path, outputs


def build_command(*, config_path, base_model, stage2_checkpoint, silence, input_root,
                  output_root, generation) -> list[str]:
    if int(generation.get("nproc_per_node", 8)) != 8:
        raise ValueError("Official SolarWM-H3 source-length inference requires single-node SP8")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc-per-node=8", "-m",
               "worldatlas.models.adapters.solarwm.solarwm_h3_worker", "infer",
               "--config", str(config_path)]
    overrides = {
        "model.checkpoint_path": str(base_model),
        "checkpoint.resume_from": str(stage2_checkpoint),
        "checkpoint.weight_source": "ema",
        "data.silence_latents_path": str(silence),
        "inference.plan": str(input_root / "plan.json"),
        "inference.dataset_root": str(input_root / "source"),
        "inference.work_dir": str(input_root.parent / "condition-cache"),
        "inference.expected_step": int(generation.get("expected_step", 3900)),
        "inference.stream_decode": bool(generation.get("stream_decode", True)),
        "validation.noise_seed": int(generation.get("seed", 42)),
        "runtime.output_dir": str(output_root),
    }
    for key, value in overrides.items():
        command.extend(["--set", _override(key, value)])
    return command


def verify_outputs(output_root: Path, outputs: dict[str, Path]) -> None:
    """Do not accept partially generated batches or unrelated completion receipts."""
    receipt = json.loads((output_root / "WORKER_COMPLETE.json").read_text())
    if set(receipt["case_keys"]) != {path.stem for path in outputs.values()}:
        raise RuntimeError("SolarWM-H3 completion does not match the batch")
    for relative in outputs.values():
        video = output_root / relative
        result = json.loads(video.with_suffix(".json").read_text())
        digest = hashlib.sha256()
        with video.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != result["output_sha256"]:
            raise RuntimeError(f"SolarWM-H3 output digest mismatch: {video}")


class CompletedPublisher:
    """Publish each official atomic result receipt, even if a later case fails."""

    def __init__(self, rows, plan, output_root, outputs, expected_step):
        self.rows, self.plan = rows, plan
        self.output_root, self.outputs = output_root, outputs
        self.expected_step = expected_step
        self.count = 0

    def __call__(self):
        # Upstream generates in plan order. Inspect only the next result rather
        # than repeatedly statting thousands of paths on shared storage.
        while self.count < len(self.rows):
            row, planned = self.rows[self.count], self.plan[self.count]
            video = self.output_root / self.outputs[row["sample_id"]]
            manifest = video.with_suffix(".json")
            if not manifest.is_file():
                return
            result = json.loads(manifest.read_text())
            if (result.get("source_row") != planned
                    or result.get("step") != self.expected_step
                    or result.get("weights_source") != "ema"):
                raise RuntimeError(f"SolarWM-H3 result identity mismatch: {manifest}")
            digest = hashlib.sha256()
            with video.open("rb") as handle:
                for block in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != result.get("output_sha256"):
                raise RuntimeError(f"SolarWM-H3 output digest mismatch: {video}")
            copy_output(video, Path(row["output_path"]))
            print_status(row["sample_id"], "generated", output_path=row["output_path"],
                         pose_source=row.get("pose_source"),
                         source_frame_count=row["source_frame_count"],
                         solarwm_run=str(self.output_root))
            self.count += 1


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    checkpoint = Path(args.checkpoint_dir).resolve()
    rows = load_batch_spec(Path(args.batch_spec_path).resolve())
    generation = load_json(Path(args.generation_config_path).resolve())
    config, base, stage2 = resolve_runtime_paths(repo, checkpoint, generation)
    silence = _resolve_child(checkpoint, generation.get("silence_latents_path"),
                             "support/h3_silence_153_158_170.safetensors")
    if not silence.is_file():
        raise FileNotFoundError(f"Missing H3 silence support artifact: {silence}")
    if not (stage2 / "COMPLETE.json").is_file() or not (stage2 / "ema.pt").is_file():
        raise FileNotFoundError(f"Incomplete SolarWM-H3 Stage2 EMA release: {stage2}")
    # Batch specs have UUID names; a suite-scoped cache lets retries reuse
    # verified conditions and staged rollouts after an interrupted run.
    root = Path(args.batch_spec_path).resolve().parents[1] / "_solarwm_h3_work"
    input_root, output_root = root / "input", root / "output"
    env = build_solarwm_environment(generation)
    world_root = Path(__file__).resolve().parents[4]
    env["PYTHONPATH"] = os.pathsep.join((str(world_root), str(repo / "src"), env.get("PYTHONPATH", "")))
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    env["WORLDATLAS_H3_DIAGNOSTICS_DIR"] = str(diagnostics)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("NCCL_DEBUG", "INFO")
    env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV")
    env.setdefault("NCCL_DEBUG_FILE", str(diagnostics / "nccl.%h.%p.log"))
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    env.setdefault("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", str(diagnostics / "nccl_trace_rank_"))
    command = build_command(config_path=config, base_model=base, stage2_checkpoint=stage2,
                            silence=silence, input_root=input_root, output_root=output_root,
                            generation=generation)
    validate_solarwm_command(command, repo, env=env)
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError("SolarWM-H3 needs exactly 8 visible GPUs for its official SP8 route")
    plan_path, outputs = build_input_bundle(rows, input_root, generation)
    publish = CompletedPublisher(rows, json.loads(plan_path.read_text()), output_root, outputs,
                                 int(generation.get("expected_step", 3900)))
    print(f"[h3-runtime] diagnostics={diagnostics}", flush=True)
    # Keep the official prepare-all -> generate-all lifecycle in one SP8
    # worker group: each rank loads the transformer and EMA only once.
    print(f"[h3-runtime] phase=full_suite samples={len(outputs)}", flush=True)
    run_supervised(command, cwd=repo, env=env, publish=publish, diagnostics=diagnostics,
                   stall_timeout_seconds=generation.get("stall_timeout_seconds", 1800))
    verify_outputs(output_root, outputs)
    publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
