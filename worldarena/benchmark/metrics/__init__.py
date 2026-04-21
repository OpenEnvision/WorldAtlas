from worldarena.benchmark.metrics.consistency import (
    OpticalFlowAepeMetric,
    ReprojectionErrorMetric,
)
from worldarena.benchmark.metrics.control import (
    CameraErrorMetric,
    MotionAccuracyMetric,
    ObjectDetectionMetric,
)
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS, group_metrics_by_dimension
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.registry import build_metrics

__all__ = [
    "CameraErrorMetric",
    "METRIC_DIMENSIONS",
    "MotionAccuracyMetric",
    "MotionMagnitudeMetric",
    "MotionSmoothnessMetric",
    "ObjectDetectionMetric",
    "OpticalFlowAepeMetric",
    "ReprojectionErrorMetric",
    "build_metrics",
    "group_metrics_by_dimension",
]
