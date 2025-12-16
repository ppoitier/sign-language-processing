from typing import Optional

from slp.config.model import MultiHeadModelConfig
from slp.config.task import TaskConfig
from slp.config.experiment import ExperimentConfig
from slp.config.data import SegmentationDatasetConfig
from slp.config.training import SegmentationTrainingConfig


class SegmentationTaskConfig(TaskConfig):
    datasets: dict[str, SegmentationDatasetConfig]
    model: MultiHeadModelConfig
    experiment: ExperimentConfig
    training: Optional[SegmentationTrainingConfig] = None
