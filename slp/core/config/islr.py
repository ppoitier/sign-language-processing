from typing import Optional

from pydantic import BaseModel
from slp.core.config.data import RecognitionDatasetConfig
from slp.core.config.experiment import ExperimentConfig
from slp.core.config.model import MultiHeadModelConfig, ContrastiveModelConfig
from slp.core.config.training import TrainingConfig


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
