from pydantic import BaseModel


class CriterionConfig(BaseModel):
    name: str
    kwargs: dict = {}
    n_classes: int = 2
    use_weights: bool = False


class TrainingConfig(BaseModel):
    criterion: CriterionConfig
    max_epochs: int
    learning_rate: float
    multi_layer_output: bool = False
    gradient_clipping: float = 0.0
    early_stopping_patience: int = 10
    debug: bool = False
