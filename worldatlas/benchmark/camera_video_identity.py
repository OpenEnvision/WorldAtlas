"""Retained videos with recorded camera intent, without claiming image pairing."""
from copy import deepcopy
from pathlib import Path
import re

from worldatlas.benchmark.integrity import digest_json

POLICY = 'retained_video_camera_intent_v1'
TOKENS = {'orbit_left', 'orbit_right', 'pan_left', 'pan_right',
          'move_left', 'move_right', 'push_in', 'pull_out'}


def recorded_camera_path(record):
    # Some camera adapters simplify the text but retain their full camera_path.
    tokens = record.get('camera_path') or [
        text.strip().replace(' ', '_')
        for text in re.findall(r'(?:^|\.\s*)Camera ([^.]+)\.', record.get('prompt') or '')]
    if not isinstance(tokens, list) or not tokens or any(t not in TOKENS for t in tokens):
        raise ValueError('Missing/unsupported recorded camera intent')
    return list(tokens)


def validate_intent(sample):
    binding = sample.prediction_binding or {}
    record = binding.get('generation_record') or {}
    if (sample.sample_id_policy != POLICY or binding.get('policy') != POLICY
            or sample.suite != 'image_static' or sample.source_sha256
            or sample.is_formal or sample.has_pose
            or record.get('sample_id') != sample.sample_id
            or record.get('suite') != sample.suite
            or record.get('status') not in {'generated', 'skipped_existing'} or record.get('error')
            or Path(record.get('prediction_path') or '').name != Path(sample.path).name
            or digest_json(record) != binding.get('generation_record_sha256')
            or sample.prompt_current or sample.prompt_sequence
            or sample.prompt_target != record.get('prompt')
            or sample.camera_path != recorded_camera_path(record)):
        raise ValueError('Invalid recorded camera/video binding')
    return record


def video_only_sample(sample):
    """Reuse strict video-byte/root checks after validating the separate intent."""
    from worldatlas.benchmark.video_metric_identity import POLICY as VIDEO_POLICY
    validate_intent(sample)
    video = deepcopy(sample)
    video.sample_id_policy = VIDEO_POLICY
    video.prediction_binding['policy'] = VIDEO_POLICY
    video.prompt_target = ''
    video.camera_path = []
    return video


def resolve_video(sample, root):
    from worldatlas.benchmark.video_metric_identity import resolve_video as resolve
    return resolve(video_only_sample(sample), root)
