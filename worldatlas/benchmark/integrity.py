"""Content fingerprints for evaluation inputs, code and per-sample checkpoints."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path

from worldatlas.benchmark.sample_identity import file_sha256


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def path_content(path: Path) -> dict:
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "missing": True}
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path)}
    return {"path": str(path), "files": {str(p.relative_to(path)): path_content(p)
            for p in sorted(path.rglob('*')) if p.is_file()}}


def sample_input_content(sample, prediction_path: Path | None, *, content_reader=None) -> dict:
    from worldatlas.benchmark.video_metric_identity import POLICY
    from worldatlas.benchmark.camera_video_identity import POLICY as CAMERA_POLICY
    if getattr(sample, 'sample_id_policy', None) in {POLICY, CAMERA_POLICY}:
        if prediction_path is None:
            raise ValueError('Video-only scoring requires a pinned prediction')
        reader = path_content if content_reader is None else content_reader
        result = {str(prediction_path): reader(prediction_path)}
        if sample.sample_id_policy == CAMERA_POLICY:
            from worldatlas.benchmark.camera_video_identity import validate_intent
            result['recorded_camera_intent'] = {'sha256': digest_json(validate_intent(sample))}
        return result
    paths = set()
    for key in ("path", "reference_path", "conditioning_path", "mask_path", "annotation_path"):
        raw = getattr(sample, key, None)
        if raw:
            paths.add(Path(str(raw).split('::', 1)[0]))
    if prediction_path is not None:
        paths.add(prediction_path)
        # Generation/action and geometry sidecars are also scoring inputs.
        from worldatlas.benchmark.generation_records import _candidate_record_files, _find_generation_record, _resolve_record_path
        # Filename-ID samples consume ONLY the content-pinned record embedded
        # in their binding (_find_generation_record below). Historical sibling
        # JSONLs are not inputs for those samples. Scanning every 5,000-video
        # directory per sample both wastes FUSE I/O and couples fingerprints to
        # unrelated legacy records. Legacy/unbound lookup retains fresh scans.
        if not getattr(sample, 'sample_id_policy', None):
            paths.update(_candidate_record_files(prediction_path))
        # Native camera/keystroke metrics may read a separately referenced
        # action JSON. Hashing only its pathname in the record is insufficient.
        record = _find_generation_record(sample, prediction_path)
        if record and not isinstance(record.get('action_payload'), dict):
            action_path = _resolve_record_path(record.get('action_spec_path'), base_dir=prediction_path.resolve().parent)
            if action_path is not None:
                paths.add(action_path)
        for name in (prediction_path.stem, prediction_path.stem + '_ann', prediction_path.stem + '_vipe_ann'):
            sidecar = prediction_path.parent / name
            if sidecar.is_dir():
                paths.add(sidecar)
    reader = path_content if content_reader is None else content_reader
    return {str(p): reader(p) for p in sorted(paths)}


def evaluation_source_hashes(project_root: Path) -> dict[str, str]:
    paths = set()
    for name in ('worldatlas/benchmark', 'worldatlas/common'):
        paths.update((project_root / name).rglob('*.py'))
    paths.add(project_root / 'scripts/run_eval_shards.py')
    # Include upstream implementations used by the two repaired metrics.
    for name in ('thirdparty/SEA-RAFT/core', 'thirdparty/SEA-RAFT/config',
                 'thirdparty/sam2/sam2', 'thirdparty/met3r/met3r'):
        for extension in ('*.py', '*.json', '*.yaml'):
            paths.update((project_root / name).rglob(extension))
    return {str(p.relative_to(project_root)): file_sha256(p) for p in sorted(paths) if p.is_file()}


def scoring_environment() -> dict:
    return {key: value for key, value in sorted(os.environ.items())
            if key.startswith(('WORLDATLAS_', 'TORCH_', 'CUBLAS_', 'CUDNN_'))
            and not any(secret in key for secret in ('TOKEN', 'PASSWORD', 'SECRET', 'API_KEY', 'AUTHORIZATION'))
            and not key.startswith(('WORLDATLAS_EVAL_', 'WORLDATLAS_JOB_', 'WORLDATLAS_BATCH_'))}


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'opencv-python', 'opencv-python-headless', 'transformers'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def checkpoint_content_for_metrics(metric_names, project_root: Path) -> dict:
    names = set(metric_names)
    root = Path(os.environ.get('WORLDATLAS_CHECKPOINT_ROOT') or project_root / 'ckpt')
    files = {}
    if names & {'motion_accuracy', 'motion_magnitude', 'optical_flow_aepe'}:
        files['sea_raft'] = Path(os.environ.get('WORLDATLAS_SEA_RAFT_CHECKPOINT') or root / 'official_metrics/Tartan-C-T-TSKH-spring540x960-M.pth')
    if 'motion_accuracy' in names:
        files['sam2'] = root / 'official_metrics/sam2.1_hiera_base_plus.pt'
    return {name: path_content(path) for name, path in files.items()}


def scoring_context(config, project_root: Path, metric_names=None) -> dict:
    if metric_names is None:
        metric_names = {name for names in config.metrics_by_suite.values() for name in names}
    return {'version': 'content_bound_v2', 'config': asdict(config) if is_dataclass(config) else vars(config),
            'package_versions': package_versions(),
            'motion_checkpoint_content': checkpoint_content_for_metrics(metric_names, project_root),
            'code': evaluation_source_hashes(project_root), 'environment': scoring_environment()}


def sample_fingerprint(sample, prediction_path: Path, context: dict) -> str:
    return digest_json({'context': context, 'sample': sample.to_dict(),
                        'inputs': sample_input_content(sample, prediction_path)})
