from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from worldarena.benchmark import run_benchmark
from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.metrics.physics import _PHYSICS_RESULT_CACHE
from worldarena.benchmark.physics_backend import (
    _decode_archived_array,
    evaluate_online_physics_i2v,
    export_pred_world_from_runtime,
)
from worldarena.benchmark.physics_manifest import (
    REQUIRED_CORE_METRICS,
    build_worldarena_physics_manifest,
)


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark_physics.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "image",
            video_root=tmp_path / "videos",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports_root",
            cache_dir=tmp_path / ".cache" / "benchmark",
        ),
        snapshot_name="physics_snapshot",
        target_fps=24,
        anchor_positions=[0.0, 0.5, 1.0],
        motion_frame_count=8,
        split_ratios={"dev_public": 0.2, "dev_private": 0.1, "test_hidden": 0.7},
        metrics_by_suite={
            "experimental": ["mse", "worldline_l2_error"],
        },
        metric_backends={
            "mse": "simulator_reference",
            "worldline_l2_error": "simulator_reference",
        },
        normalization={
            "mse": {"lower": 0.0, "upper": 1.0, "higher_is_better": False},
            "worldline_l2_error": {"lower": 0.0, "upper": 2.0, "higher_is_better": False},
        },
    )


def _write_video(path: Path, *, width: int = 32, height: int = 24, num_frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (width, height),
    )
    for index in range(num_frames):
        frame = np.full((height, width, 3), fill_value=40 * (index + 1), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_png(path: Path, *, width: int = 32, height: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 180
    cv2.imwrite(str(path), image)


def _write_runtime_raw(
    runtime_data_root: Path,
    *,
    sample_id: str = "0000",
    num_frames: int = 3,
) -> None:
    raw_root = runtime_data_root / "OUT_ViPE_Raw"
    for name in ["depth", "mask", "pose", "intrinsics"]:
        (raw_root / name).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(raw_root / "depth" / f"{sample_id}.zip", "w") as archive:
        for index in range(num_frames):
            buffer = io.BytesIO()
            np.save(buffer, np.full((12, 16), fill_value=index + 1, dtype=np.float32))
            archive.writestr(f"{index:05d}.npy", buffer.getvalue())

    with zipfile.ZipFile(raw_root / "mask" / f"{sample_id}.zip", "w") as archive:
        for _ in range(num_frames):
            buffer = io.BytesIO()
            np.save(buffer, np.ones((12, 16), dtype=np.uint8))
            archive.writestr("frame.npy", buffer.getvalue())

    (raw_root / "mask" / f"{sample_id}.txt").write_text(
        "0: background\n1: ball\n",
        encoding="utf-8",
    )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    np.savez(raw_root / "pose" / f"{sample_id}.npz", data=poses, inds=np.arange(num_frames))
    intrinsics = np.repeat(
        np.asarray([[20.0, 20.0, 8.0, 6.0]], dtype=np.float32),
        num_frames,
        axis=0,
    )
    np.savez(raw_root / "intrinsics" / f"{sample_id}.npz", data=intrinsics, inds=np.arange(num_frames))
    metadata_root = runtime_data_root / "INPUT_DATA" / "Metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / f"{sample_id}.json").write_text(
        json.dumps(
            {
                "conditioning_tracks": ["i2v"],
                "video_prompt": "A toy ball bounces on a tabletop.",
                "fg_prompt": "toy ball",
                "primary": "ball",
                "secondary": "",
                "primary_obj_rot_axis": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fake_physics_pipeline(fake_root: Path) -> None:
    fake_root.mkdir(parents=True, exist_ok=True)
    metric_payload = {metric: 0.1 for metric in REQUIRED_CORE_METRICS}
    metric_payload["mse"] = 0.05
    metric_payload["worldline_l2_error"] = 0.2

    (fake_root / "main_part2.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "import io",
                "import os",
                "import zipfile",
                "from pathlib import Path",
                "import numpy as np",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--video', required=True)",
                "args = parser.parse_args()",
                "data_root = Path(os.environ['WORLDARENA_PHYSICS_DATA_ROOT'])",
                "raw_root = data_root / 'OUT_ViPE_Raw'",
                "for name in ['depth', 'mask', 'pose', 'intrinsics']:",
                "    (raw_root / name).mkdir(parents=True, exist_ok=True)",
                "with zipfile.ZipFile(raw_root / 'depth' / f'{args.video}.zip', 'w') as archive:",
                "    for index in range(3):",
                "        buffer = io.BytesIO()",
                "        np.save(buffer, np.full((12, 16), fill_value=index + 1, dtype=np.float32))",
                "        archive.writestr(f'{index:05d}.npy', buffer.getvalue())",
                "with zipfile.ZipFile(raw_root / 'mask' / f'{args.video}.zip', 'w') as archive:",
                "    for index in range(3):",
                "        buffer = io.BytesIO()",
                "        np.save(buffer, np.ones((12, 16), dtype=np.uint8))",
                "        archive.writestr(f'{index:05d}.npy', buffer.getvalue())",
                "(raw_root / 'mask' / f'{args.video}.txt').write_text('0: background\\n1: ball\\n', encoding='utf-8')",
                "poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)",
                "np.savez(raw_root / 'pose' / f'{args.video}.npz', data=poses, inds=np.arange(3))",
                "intrinsics = np.asarray([[20.0, 20.0, 8.0, 6.0]] * 3, dtype=np.float32)",
                "np.savez(raw_root / 'intrinsics' / f'{args.video}.npz', data=intrinsics, inds=np.arange(3))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (fake_root / "benchmark_export.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "import json",
                "import os",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--benchmark-name', required=True)",
                "parser.add_argument('--sample-ids', nargs='+', required=True)",
                "parser.add_argument('--render-split', required=True)",
                "parser.add_argument('--overwrite', action='store_true')",
                "args = parser.parse_args()",
                "data_root = Path(os.environ['WORLDARENA_PHYSICS_DATA_ROOT'])",
                "world_gt = data_root / 'OUT_Benchmark' / args.benchmark_name / 'cases' / args.sample_ids[0] / 'world_gt'",
                "world_gt.mkdir(parents=True, exist_ok=True)",
                "for filename in ['rgb.mp4', 'depth.npz', 'flow.npz', 'mask.npz', 'occlusion.npz', 'cameras.npz', 'points4d.npz', 'worldlines.npz']:",
                "    (world_gt / filename).write_bytes(b'ok')",
                "(world_gt / 'export_info.json').write_text(json.dumps({'ok': True}), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (fake_root / "benchmark_eval.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "import json",
                "from pathlib import Path",
                "counter_path = Path(__file__).with_name('benchmark_eval_calls.txt')",
                "count = int(counter_path.read_text(encoding='utf-8')) if counter_path.exists() else 0",
                "counter_path.write_text(str(count + 1), encoding='utf-8')",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--manifest', type=Path, required=True)",
                "parser.add_argument('--pred-dir', type=Path, required=True)",
                "parser.add_argument('--pred-world-root', type=Path, default=None)",
                "parser.add_argument('--output', type=Path, required=True)",
                "parser.add_argument('--skip-rgb-warp-lpips', action='store_true')",
                "args = parser.parse_args()",
                "assert args.pred_world_root is not None",
                "pred_world_dirs = [path for path in args.pred_world_root.iterdir() if path.is_dir()]",
                "assert pred_world_dirs, 'pred world directory missing'",
                "assert (pred_world_dirs[0] / 'depth.npz').exists()",
                f"metrics = {metric_payload!r}",
                "per_video = {'stem': pred_world_dirs[0].name, 'frame_count_match': True, 'frame_coverage': 1.0, 'missing_world_artifacts': []}",
                "per_video.update(metrics)",
                "payload = {'summary': {'num_videos': 1}, 'per_video': [per_video]}",
                "args.output.parent.mkdir(parents=True, exist_ok=True)",
                "args.output.write_text(json.dumps(payload, indent=2) + '\\n', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_sample(reference_path: Path, physics_root: Path, output_root: Path) -> BenchmarkSample:
    case_root = output_root / "cases" / "physics__0000__i2v"
    simulator_root = case_root / "simulator"
    world_gt_root = case_root / "world_gt"
    simulator_root.mkdir(parents=True, exist_ok=True)
    world_gt_root.mkdir(parents=True, exist_ok=True)
    metadata_template = simulator_root / "metadata_template.json"
    metadata_template.write_text(
        json.dumps(
            {
                "conditioning_tracks": ["i2v"],
                "video_prompt": "A toy ball bounces on a tabletop.",
                "fg_prompt": "toy ball",
                "primary": "ball",
                "secondary": "",
                "primary_obj_rot_axis": None,
                "submitted_video_path": str(simulator_root / "submitted_video.mp4"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return BenchmarkSample(
        sample_id="physics__0000__i2v",
        suite="experimental",
        split="dev_public",
        asset_level="video_nonformal",
        modality="video",
        is_formal=False,
        eligible_for_official=False,
        category_path="physics/i2v",
        path=str(reference_path),
        relative_path="physics/assets/0000.mp4",
        reference_path=str(reference_path),
        prediction_stem="physics__0000__i2v",
        conditioning_strategy="external_image",
        prompt_current="A toy ball bounces on a tabletop.",
        prompt_target="A toy ball bounces on a tabletop.",
        style=None,
        environment=None,
        scene=None,
        motion_category="ball",
        source_name="worldarena_physics",
        source_type="simulator_video",
        source_group_id="physics::0000",
        license_bucket="internal",
        has_annotation=True,
        has_instruction=True,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=32,
        height=24,
        fps=8.0,
        duration_seconds=0.375,
        duration_bucket=None,
        track="physics_i2v",
        conditioning_path=str(reference_path),
        physics_spec={
            "metric_contract": {
                "required_core_metrics": list(REQUIRED_CORE_METRICS),
                "require_full_frame_coverage": True,
                "require_pred_world_artifacts": True,
            }
        },
        physics_case={
            "physics_pipeline_root": str(physics_root),
            "physics_data_root": str((physics_root / "data_root").resolve()),
            "source_sample_id": "0000",
            "benchmark_sample_id": "physics__0000__i2v",
            "case_root": str(case_root.resolve()),
            "simulator_root": str(simulator_root.resolve()),
            "submitted_video_path": str((simulator_root / "submitted_video.mp4").resolve()),
            "metadata_template_path": str(metadata_template.resolve()),
            "world_gt_root": str(world_gt_root.resolve()),
            "physics_export_benchmark_name": "worldarena_physics",
            "physics_render_split": "original_length",
        },
    )


def test_build_worldarena_physics_manifest_roundtrip(tmp_path: Path) -> None:
    physics_pipeline_root = tmp_path / "benchmarks" / "WorldEval-X" / "physics_pipeline" / "worldarena_physics"
    assets_root = physics_pipeline_root / "assets"
    video_path = assets_root / "0000.mp4"
    runtime_data_root = tmp_path / "benchmarks" / "WorldEval-X" / ".runtime" / "physics_pipeline" / "data_root"
    _write_video(video_path)
    runtime_data_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "0000.json").write_text(
        json.dumps(
            {
                "video_prompt": "A toy ball bounces on a tabletop.",
                "fg_prompt": "toy ball",
                "primary": "ball",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_worldarena_physics_manifest(
        physics_pipeline_root=physics_pipeline_root,
        output_root=tmp_path / "physics_benchmark",
        sample_ids=["0000"],
    )

    assert result["items"] == 1
    manifest = result["manifest"]
    assert len(manifest) == 1
    sample = manifest[0]
    assert sample.sample_id == "physics__0000__i2v"
    assert sample.suite == "experimental"
    assert sample.track == "physics_i2v"
    assert sample.conditioning_path is not None
    assert Path(sample.conditioning_path).is_file()
    assert sample.physics_case is not None
    assert sample.physics_spec is not None
    assert sample.physics_case["physics_data_root"] == str(runtime_data_root.resolve())
    assert Path(result["manifest_path"]).is_file()


def test_build_worldarena_physics_manifest_honors_data_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics_pipeline_root = tmp_path / "benchmarks" / "WorldEval-X" / "physics_pipeline" / "worldarena_physics"
    assets_root = physics_pipeline_root / "assets"
    override_data_root = tmp_path / "custom_runtime" / "data_root"
    video_path = assets_root / "0000.mp4"
    _write_video(video_path)
    override_data_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "0000.json").write_text(
        json.dumps(
            {
                "video_prompt": "A toy ball bounces on a tabletop.",
                "fg_prompt": "toy ball",
                "primary": "ball",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLDARENA_PHYSICS_DATA_ROOT", str(override_data_root))

    result = build_worldarena_physics_manifest(
        physics_pipeline_root=physics_pipeline_root,
        output_root=tmp_path / "physics_benchmark",
        sample_ids=["0000"],
    )

    sample = result["manifest"][0]
    assert sample.physics_case is not None
    assert sample.physics_case["physics_data_root"] == str(override_data_root.resolve())


def test_export_pred_world_from_runtime_writes_required_files(tmp_path: Path) -> None:
    runtime_data_root = tmp_path / "runtime" / "data_root"
    generated_video = tmp_path / "pred.mp4"
    _write_video(generated_video, num_frames=3)
    _write_runtime_raw(runtime_data_root)

    output_root = tmp_path / "pred_world" / "sample0"
    payload = export_pred_world_from_runtime(
        sample_id="0000",
        runtime_data_root=runtime_data_root,
        generated_video_path=generated_video,
        output_root=output_root,
    )

    assert payload["success"] is True
    assert (output_root / "depth.npz").is_file()
    assert (output_root / "mask.npz").is_file()
    assert (output_root / "flow.npz").is_file()
    assert (output_root / "cameras.npz").is_file()


def test_decode_archived_array_reads_openexr_depth_channel(tmp_path: Path) -> None:
    OpenEXR = pytest.importorskip("OpenEXR")
    Imath = pytest.importorskip("Imath")

    values = np.asarray([[1.0, 2.5], [3.75, 4.0]], dtype=np.float32)
    exr_path = tmp_path / "depth.exr"
    header = OpenEXR.Header(values.shape[1], values.shape[0])
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {"Z": Imath.Channel(pixel_type)}
    writer = OpenEXR.OutputFile(str(exr_path), header)
    writer.writePixels({"Z": values.tobytes()})
    writer.close()

    decoded = _decode_archived_array(exr_path.read_bytes(), suffix=".exr")

    assert decoded.dtype == np.float32
    assert decoded.shape == values.shape
    assert np.allclose(decoded, values)


def test_evaluate_online_physics_i2v_runs_fake_pipeline(tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_physics_pipeline"
    _write_fake_physics_pipeline(fake_root)
    ref_image = tmp_path / "ref.png"
    generated_video = tmp_path / "generated.mp4"
    _write_png(ref_image)
    _write_video(generated_video)

    result = evaluate_online_physics_i2v(
        prompt="A toy ball bounces on a tabletop.",
        image_path=ref_image,
        generated_video_path=generated_video,
        output_root=tmp_path / "online_eval",
        physics_pipeline_root=fake_root,
    )

    assert result["success"] is True
    assert result["mse"] == 0.05
    assert result["worldline_l2_error"] == 0.2
    assert result["pred_world_status"]["ready"] is True
    assert result["world_gt_status"]["full_metrics_ready"] is True


def test_runner_caches_single_physics_eval_across_multiple_metrics(tmp_path: Path) -> None:
    _PHYSICS_RESULT_CACHE.clear()
    fake_root = tmp_path / "fake_physics_pipeline"
    _write_fake_physics_pipeline(fake_root)
    reference_video = tmp_path / "reference.mp4"
    prediction_root = tmp_path / "predictions"
    prediction_path = prediction_root / "physics__0000__i2v.mp4"
    output_dir = tmp_path / "reports"
    _write_video(reference_video)
    _write_video(prediction_path)

    config = _make_config(tmp_path)
    manifest = [_make_sample(reference_video, fake_root, tmp_path / "physics_eval")]

    result = run_benchmark(
        config=config,
        manifest=manifest,
        predictions_root=prediction_root,
        output_dir=output_dir,
        model_name="toy_model",
        suites={"experimental"},
        splits={"dev_public"},
        limit=None,
    )

    assert result["run_manifest"]["missing_predictions"] == 0
    assert result["aggregate_payload"]["per_metric_summary"]["experimental"]["mse"] is not None
    assert result["aggregate_payload"]["per_metric_summary"]["experimental"]["worldline_l2_error"] is not None
    counter_path = fake_root / "benchmark_eval_calls.txt"
    assert counter_path.read_text(encoding="utf-8") == "1"
