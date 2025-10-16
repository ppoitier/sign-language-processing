from typing import Optional

from slp.config.templates.model import MultiHeadConfig
from slp.config.templates.task import TaskConfig
from slp.config.templates.experiment import ExperimentConfig
from slp.config.templates.data import SegmentationDatasetConfig
from slp.config.templates.training import SegmentationTrainingConfig


class SegmentationTaskConfig(TaskConfig):
    datasets: dict[str, SegmentationDatasetConfig]
    model: MultiHeadConfig
    experiment: ExperimentConfig
    training: Optional[SegmentationTrainingConfig] = None
