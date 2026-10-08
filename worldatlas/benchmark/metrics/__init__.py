"""Public exports for benchmark metric classes and registry helpers."""

from worldatlas.benchmark.metrics.appearance import (
    BrightnessConsistencyMetric,
    BrightnessDistributionConsistencyMetric,
    ColorTemperatureConstraintMetric,
    ColorTemperatureConsistencyMetric,
    SharpnessRetentionMetric,
    SharpnessVectorRetentionMetric,
)
from worldatlas.benchmark.metrics.background_consistency import BackgroundConsistencyMetric
from worldatlas.benchmark.metrics.consistency import (
    OpticalFlowAepeMetric,
    ReprojectionErrorMetric,
)
from worldatlas.benchmark.metrics.control import (
    CameraErrorMetric,
    MotionAccuracyMetric,
)
from worldatlas.benchmark.metrics.catalog import METRIC_DIMENSIONS, group_metrics_by_dimension
from worldatlas.benchmark.metrics.met3r import Met3RConsistencyMetric
from worldatlas.benchmark.metrics.memory import MemoryRevisitConsistencyMetric, MemorySymmetryMetric
from worldatlas.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldatlas.benchmark.metrics.quality import HPSv3NormMetric, ImageQualityMetric
from worldatlas.benchmark.metrics.reconstruction import (
    ReconstructionConsistencyEvaluator,
    ReconstructionConsistencyMetric,
)
from worldatlas.benchmark.metrics.segment_continuity import SegmentContinuityMetric
from worldatlas.benchmark.metrics.temporal import TemporalCalibrationMetric
from worldatlas.benchmark.metrics.temporal_flickering import TemporalFlickeringMetric
from worldatlas.benchmark.metrics.long_sequence import LongSequenceMetric
from worldatlas.benchmark.metrics.registry import build_metrics

__all__ = [
    "BackgroundConsistencyMetric",
    "BrightnessConsistencyMetric",
    "BrightnessDistributionConsistencyMetric",
    "CameraErrorMetric",
    "ColorTemperatureConstraintMetric",
    "ColorTemperatureConsistencyMetric",
    "HPSv3NormMetric",
    "ImageQualityMetric",
    "METRIC_DIMENSIONS",
    "MemoryRevisitConsistencyMetric",
    "MemorySymmetryMetric",
    "Met3RConsistencyMetric",
    "MotionAccuracyMetric",
    "MotionMagnitudeMetric",
    "MotionSmoothnessMetric",
    "OpticalFlowAepeMetric",
    "ReconstructionConsistencyMetric",
    "ReconstructionConsistencyEvaluator",
    "ReprojectionErrorMetric",
    "SegmentContinuityMetric",
    "SharpnessRetentionMetric",
    "SharpnessVectorRetentionMetric",
    "TemporalCalibrationMetric",
    "TemporalFlickeringMetric",
    "LongSequenceMetric",
    "build_metrics",
    "group_metrics_by_dimension",
]
