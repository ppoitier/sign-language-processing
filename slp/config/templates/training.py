from typing import Optional

from pydantic import BaseModel

from slp.config.templates.codec import SegmentCodecConfig


class CriterionConfig(BaseModel):
    name: str
    kwargs: dict = {}
    n_classes: int = 2
    use_weights: bool = False


class TrainingConfig(BaseModel):
    criterion: dict[str, CriterionConfig]
    max_epochs: int
    learning_rate: float
    is_output_multilayer: bool = False
    gradient_clipping: float = 0.0
    early_stopping_patience: int = 10
    checkpoint_path: Optional[str] = None
    n_classes: Optional[int] = None


class SegmentationTrainingConfig(TrainingConfig):
    use_offsets: bool = False
    heads_to_targets: dict[str, str]
    segment_decoder: SegmentCodecConfig
