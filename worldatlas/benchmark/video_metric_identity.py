"""Content-bound video-only scoring, explicitly not an image/task binding."""
from pathlib import Path
from worldatlas.benchmark.sample_identity import file_sha256

POLICY = 'retained_video_metrics_v1'
# AEPE uses prediction.path; Style compares the prediction's own anchor frames.
METRICS = frozenset({'met3r_consistency','geometric_consistency','photometric_consistency','jedi','fvmd','trajan','optical_flow_aepe','style_consistency'})


def validate_metric_scope(sample, metric_names, require_image_bindings):
    from worldatlas.benchmark.camera_video_identity import POLICY as CAMERA_POLICY, validate_intent
    if sample.sample_id_policy == CAMERA_POLICY:
        if require_image_bindings or set(metric_names or []) != {'camera_error'}:
            raise ValueError('Recorded camera video inputs are restricted to camera_error')
        validate_intent(sample)
        return
    if sample.sample_id_policy != POLICY:
        return
    if require_image_bindings or not metric_names or not set(metric_names) <= METRICS:
        raise ValueError('Video-only inputs are restricted to image/mask-independent metrics')


def resolve_video(sample, root):
    binding = sample.prediction_binding or {}
    if (sample.sample_id_policy != POLICY or sample.modality != 'video'
            or sample.source_name != 'retained_prediction_video_roster_v1'
            or sample.suite not in {'image_static','image_dynamic'}
            or sample.has_mask or sample.mask_path or sample.annotation_path or sample.conditioning_path
            or sample.prompt_current or sample.prompt_target or sample.camera_path or sample.prompt_sequence):
        raise ValueError('Video-only inputs cannot assert image, mask, text or control provenance')
    if binding.get('policy') != POLICY or binding.get('sample_id') != sample.sample_id:
        raise ValueError('Wrong video-only binding')
    root = Path(root).resolve()
    if Path(binding.get('predictions_root','')).resolve() != root:
        raise ValueError('Wrong model video root')
    path = Path(binding.get('path',''))
    if not path.is_absolute() or path.resolve().parent != root:
        raise ValueError('Video must belong to the declared model root')
    if (Path(sample.path).resolve() != path.resolve() or Path(sample.reference_path).resolve() != path.resolve()
            or path.stem != sample.sample_id or sample.prediction_stem != sample.sample_id):
        raise ValueError('Video-only roster must identify the actual video, not an input image')
    if not path.is_file() or not path.stat().st_size or file_sha256(path) != binding.get('prediction_sha256'):
        raise ValueError('Video missing, empty or changed')
    return path
