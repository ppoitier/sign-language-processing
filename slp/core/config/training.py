from typing import Optional

from pydantic import BaseModel, Field


class CriterionConfig(BaseModel):
    name: str
    kwargs: dict = {}
    n_classes: Optional[int] = None
    use_weights: bool = False
    weight_strategy: str = "inverse"
    multi_layer: bool = False


class LRSchedulerConfig(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)
    monitor: Optional[str] = None


class TrainingConfig(BaseModel):
    max_epochs: int
    n_warmup_epochs: Optional[int] = None
    lr_scheduler: Optional[LRSchedulerConfig] = None

    loss_functions: dict[str, CriterionConfig]
    learning_rate: float
    early_stopping_patience: int = 10
    gradient_clipping: float = 0.0
    overfit_one_batch: bool = False

    checkpoint_path: Optional[str] = None
    n_classes: Optional[int] = None
    is_output_multistage: bool = False
    heads_to_targets: dict[str, str]


class SegmentDecoderConfig(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)


class SegmentationTrainingConfig(TrainingConfig):
    use_offsets: bool = False
    segment_decoder: SegmentDecoderConfig
