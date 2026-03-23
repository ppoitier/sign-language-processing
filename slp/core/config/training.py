from typing import Optional

from pydantic import BaseModel

from slp.core.config.codec import SegmentCodecConfig


class CriterionConfig(BaseModel):
    name: str
    kwargs: dict = {}
    n_classes: Optional[int] = None
    use_weights: bool = False
    multi_layer: bool = False


class TrainingConfig(BaseModel):
    max_epochs: int
    n_warmup_epochs: Optional[int] = None

    loss_functions: dict[str, CriterionConfig]
    learning_rate: float
    early_stopping_patience: int = 10
    gradient_clipping: float = 0.0
    overfit_one_batch: bool = False

    checkpoint_path: Optional[str] = None
    n_classes: Optional[int] = None
    is_output_multilayer: bool = False


class SegmentationTrainingConfig(TrainingConfig):
    use_offsets: bool = False
    heads_to_targets: dict[str, str]
    segment_decoder: SegmentCodecConfig
