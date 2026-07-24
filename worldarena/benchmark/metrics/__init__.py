"""Public exports for benchmark metric classes and registry helpers."""

from worldarena.benchmark.metrics.appearance import (
    BrightnessConsistencyMetric,
    BrightnessDistributionConsistencyMetric,
    ColorTemperatureConstraintMetric,
    ColorTemperatureConsistencyMetric,
    SharpnessRetentionMetric,
    SharpnessVectorRetentionMetric,
)
from worldarena.benchmark.metrics.background_consistency import BackgroundConsistencyMetric
from worldarena.benchmark.metrics.consistency import (
    OpticalFlowAepeMetric,
    ReprojectionErrorMetric,
)
from worldarena.benchmark.metrics.control import (
    CameraErrorMetric,
    MotionAccuracyMetric,
    ObjectDetectionMetric,
)
from worldarena.benchmark.metrics.depth_collision import DepthCollisionMetric
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS, group_metrics_by_dimension
from worldarena.benchmark.metrics.legacy import (
    DepthAccuracyMetric,
    SemanticAlignmentMetric,
)
from worldarena.benchmark.metrics.met3r import Met3RConsistencyMetric
from worldarena.benchmark.metrics.memory import MemoryRevisitConsistencyMetric, MemorySymmetryMetric
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.physics_judge import PhysicsJudgeMetric
from worldarena.benchmark.metrics.pmf import PhysicalMotionFidelityMetric
from worldarena.benchmark.metrics.quality import HPSv3NormMetric, ImageQualityMetric
from worldarena.benchmark.metrics.reconstruction import (
    ReconstructionConsistencyEvaluator,
    ReconstructionConsistencyMetric,
)
from worldarena.benchmark.metrics.regions import (
    ObjectPermanenceMetric,
    SeenRegionPreservationMetric,
)
from worldarena.benchmark.metrics.segment_continuity import SegmentContinuityMetric
from worldarena.benchmark.metrics.temporal import TemporalCalibrationMetric
from worldarena.benchmark.metrics.temporal_flickering import TemporalFlickeringMetric
from worldarena.benchmark.metrics.long_sequence import LongSequenceMetric
from worldarena.benchmark.metrics.registry import build_metrics

__all__ = [
    "BackgroundConsistencyMetric",
    "BrightnessConsistencyMetric",
    "BrightnessDistributionConsistencyMetric",
    "CameraErrorMetric",
    "ColorTemperatureConstraintMetric",
    "ColorTemperatureConsistencyMetric",
    "DepthAccuracyMetric",
    "DepthCollisionMetric",
    "HPSv3NormMetric",
    "ImageQualityMetric",
    "METRIC_DIMENSIONS",
    "MemoryRevisitConsistencyMetric",
    "MemorySymmetryMetric",
    "Met3RConsistencyMetric",
    "MotionAccuracyMetric",
    "MotionMagnitudeMetric",
    "MotionSmoothnessMetric",
    "ObjectDetectionMetric",
    "ObjectPermanenceMetric",
    "PhysicalMotionFidelityMetric",
    "OpticalFlowAepeMetric",
    "PhysicsJudgeMetric",
    "ReconstructionConsistencyMetric",
    "ReconstructionConsistencyEvaluator",
    "ReprojectionErrorMetric",
    "SeenRegionPreservationMetric",
    "SegmentContinuityMetric",
    "SemanticAlignmentMetric",
    "SharpnessRetentionMetric",
    "SharpnessVectorRetentionMetric",
    "TemporalCalibrationMetric",
    "TemporalFlickeringMetric",
    "LongSequenceMetric",
    "build_metrics",
    "group_metrics_by_dimension",
]
