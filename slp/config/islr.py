from typing import Optional

from pydantic import BaseModel
from slp.config.data import RecognitionDatasetConfig
from slp.config.experiment import ExperimentConfig
from slp.config.model import MultiHeadModelConfig, ContrastiveModelConfig
from slp.config.training import TrainingConfig


class IsolatedRecognitionTaskConfig(BaseModel):
    datasets: dict[str, RecognitionDatasetConfig]
    model: MultiHeadModelConfig
    experiment: ExperimentConfig
    training: Optional[TrainingConfig] = None


class ContrastiveIsolatedRecognitionTaskConfig(BaseModel):
    datasets: dict[str, RecognitionDatasetConfig]
    model: ContrastiveModelConfig
    experiment: ExperimentConfig
    contrastive_training: TrainingConfig
    linear_evaluation_training: TrainingConfig
    skip_contrastive_training: bool = False
