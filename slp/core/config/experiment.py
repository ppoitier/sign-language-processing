from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from slp.core.config.dataset import ContinuousDatasetConfig
from slp.core.config.training import (
    RepresentationLearningTrainingConfig,
    SegmentationTrainingConfig,
    TrainingConfig,
)
from slp.core.config.model import HydraConfig, VacModelConfig


class ExperimentConfig(BaseModel):
    id: str
    variant: str = 'default'

    output_dir: str
    seed: int | Literal['random'] = 'random'
    experiment_datetime: datetime = Field(default_factory=datetime.now)

    mlflow_uri: str | None = None
    show_progress_bar: bool = False
    debug: bool = False


class TaskConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    experiment: ExperimentConfig


class IsolatedRecognitionClassificationTaskConfig(TaskConfig):
    datasets: dict[str, ContinuousDatasetConfig]
    model: HydraConfig
    training: Optional[TrainingConfig] = None


class SegmentationTaskConfig(TaskConfig):
    datasets: dict[str, ContinuousDatasetConfig]
    model: HydraConfig
    training: Optional[SegmentationTrainingConfig] = None


class RepresentationLearningTaskConfig(TaskConfig):
    """Self-supervised pretraining of a backbone, task-agnostic by design."""

    datasets: dict[str, ContinuousDatasetConfig]
    model: HydraConfig
    training: Optional[RepresentationLearningTrainingConfig] = None


class ContinuousRecognitionTaskConfig(TaskConfig):
    datasets: dict[str, ContinuousDatasetConfig]
    model: VacModelConfig
    training: Optional[TrainingConfig] = None
