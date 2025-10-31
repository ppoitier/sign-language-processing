from typing import Optional

from pydantic import BaseModel
from slp.config.templates.data import RecognitionDatasetConfig
from slp.config.templates.experiment import ExperimentConfig
from slp.config.templates.model import MultiHeadModelConfig
from slp.config.templates.training import TrainingConfig


class IsolatedRecognitionTaskConfig(BaseModel):
    datasets: dict[str, RecognitionDatasetConfig]
    model: MultiHeadModelConfig
    experiment: ExperimentConfig
    training: Optional[TrainingConfig] = None
