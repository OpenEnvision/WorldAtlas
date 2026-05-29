from worldarena.benchmark.metrics.appearance import (
    BrightnessConsistencyMetric,
    BrightnessDistributionConsistencyMetric,
    ColorTemperatureConstraintMetric,
    ColorTemperatureConsistencyMetric,
    SharpnessRetentionMetric,
    SharpnessVectorRetentionMetric,
)
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
    PsnrMetric,
    SemanticAlignmentMetric,
    SsimMetric,
)
from worldarena.benchmark.metrics.met3r import Met3RConsistencyMetric
from worldarena.benchmark.metrics.memory import MemoryRevisitConsistencyMetric, MemorySymmetryMetric
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.physics_judge import PhysicsJudgeMetric
from worldarena.benchmark.metrics.pmf import PhysicalMotionFidelityMetric
from worldarena.benchmark.metrics.quality import ImageQualityMetric
from worldarena.benchmark.metrics.reconstruction import ReconstructionConsistencyMetric
from worldarena.benchmark.metrics.regions import (
    NovelRegionFidelityMetric,
    ObjectPermanenceMetric,
    SeenRegionPreservationMetric,
)
from worldarena.benchmark.metrics.temporal import TemporalCalibrationMetric
from worldarena.benchmark.metrics.long_sequence import LongSequenceMetric
from worldarena.benchmark.metrics.registry import build_metrics

__all__ = [
    "BrightnessConsistencyMetric",
    "BrightnessDistributionConsistencyMetric",
    "CameraErrorMetric",
    "ColorTemperatureConstraintMetric",
    "ColorTemperatureConsistencyMetric",
    "DepthAccuracyMetric",
    "DepthCollisionMetric",
    "ImageQualityMetric",
    "METRIC_DIMENSIONS",
    "MemoryRevisitConsistencyMetric",
    "MemorySymmetryMetric",
    "Met3RConsistencyMetric",
    "MotionAccuracyMetric",
    "MotionMagnitudeMetric",
    "MotionSmoothnessMetric",
    "NovelRegionFidelityMetric",
    "ObjectDetectionMetric",
    "ObjectPermanenceMetric",
    "PhysicalMotionFidelityMetric",
    "OpticalFlowAepeMetric",
    "PhysicsJudgeMetric",
    "PsnrMetric",
    "ReconstructionConsistencyMetric",
    "ReprojectionErrorMetric",
    "SeenRegionPreservationMetric",
    "SemanticAlignmentMetric",
    "SharpnessRetentionMetric",
    "SharpnessVectorRetentionMetric",
    "SsimMetric",
    "TemporalCalibrationMetric",
    "LongSequenceMetric",
    "build_metrics",
    "group_metrics_by_dimension",
]
