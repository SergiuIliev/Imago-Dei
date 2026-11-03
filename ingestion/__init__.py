"""
Ingestion subsystem for Personal AI Time Capsule.
Provides MediaIngestionPipeline for audiovisual assets and
ModelIngestionPipeline for modern open-weights language models.
"""

from ingestion.pipeline import MediaIngestionPipeline
from ingestion.model_pipeline import (
    ModelIngestionPipeline,
    HardwareProfiler,
    HardwareProfile,
    VRAMCalculator,
    MemoryFitReport,
    MemoryFitStatus,
    CuratedModelSpec,
    DiscoveredModel,
    ModelBenchmarkResult,
    ModelBenchmarkSuite,
    ModelRegistry,
    CURATED_MODELS
)

__all__ = [
    "MediaIngestionPipeline",
    "ModelIngestionPipeline",
    "HardwareProfiler",
    "HardwareProfile",
    "VRAMCalculator",
    "MemoryFitReport",
    "MemoryFitStatus",
    "CuratedModelSpec",
    "DiscoveredModel",
    "ModelBenchmarkResult",
    "ModelBenchmarkSuite",
    "ModelRegistry",
    "CURATED_MODELS"
]
