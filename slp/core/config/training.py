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
    kwargs: dict = {}
    monitor: Optional[str] = None


class BaseTrainingConfig(BaseModel):
    """Fields every training run needs, and the only ones ``run_training`` reads."""

    max_epochs: int
    lr_scheduler: Optional[LRSchedulerConfig] = None

    learning_rate: float
    early_stopping_patience: int = 10
    gradient_clipping: float = 0.0
    overfit_one_batch: bool = False

    skip_training: bool = False
    checkpoint_path: Optional[str] = None


class TrainingConfig(BaseTrainingConfig):
    """Supervised training: one criterion per head, each mapped to a target."""

    loss_functions: dict[str, CriterionConfig]

    n_classes: Optional[int] = None
    is_output_multistage: bool = False
    heads_to_targets: dict[str, str]


class SegmentDecoderConfig(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)


class SegmentationTrainingConfig(TrainingConfig):
    use_offsets: bool = False
    segment_target: str = 'segments'
    segment_decoder: SegmentDecoderConfig


class SSLMethodConfig(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)


class RepresentationLearningTrainingConfig(BaseTrainingConfig):
    """Self-supervised pretraining.

    Has no ``loss_functions`` or ``heads_to_targets``: the objective is the
    method, and there are no labels to map heads onto.
    """

    method: SSLMethodConfig
    embedding_head: str = "projection"
    is_output_multistage: bool = False

    log_representation_stats: bool = True
    cache_test_embeddings: bool = False
