from typing import Any, Optional, OrderedDict, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: Optional[str] = None


class HeadConfig(ModelConfig):
    n_channels: int


class HydraConfig(ModelConfig):
    backbone: ModelConfig
    neck: Optional[ModelConfig] = None
    heads: OrderedDict[str, HeadConfig]
    channels_to_head_routing: Literal["split", "shared"] = "split"
    multi_layer: bool = True
    loss_on_all_stages: bool = True


class ContrastiveModelConfig(BaseModel):
    backbone: ModelConfig
    projector: ModelConfig
    linear_evaluation_head: HeadConfig
