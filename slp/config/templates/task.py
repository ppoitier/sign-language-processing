from uuid import uuid4
from datetime import datetime

from pydantic import BaseModel, Field

from slp.config.templates.data import SegmentationDatasetConfig, SegmentCodecConfig, RecognitionDatasetConfig
from slp.config.templates.module import ModuleConfig
from slp.config.templates.training import TrainingConfig


class TaskConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    prefix: str = "experiment"
    seed: int = 42
    task_datetime: datetime = Field(default_factory=datetime.now)
    output_dir: str


class SegmentationTaskConfig(TaskConfig):
    target_codec: SegmentCodecConfig
    datasets: dict[str, SegmentationDatasetConfig]
    backbone: ModuleConfig
    training: TrainingConfig


class ContrastiveRecognitionTask(TaskConfig):
    datasets: dict[str, RecognitionDatasetConfig]
    backbone: ModuleConfig
    projector: ModuleConfig
    training: TrainingConfig
