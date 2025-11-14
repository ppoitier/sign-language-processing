from typing import Optional, Union

from pydantic import BaseModel


class ModelConfig(BaseModel):
    type: str
    args: dict = dict()
    checkpoint_path: Optional[str] = None


class HeadConfig(ModelConfig):
    in_channels_range: Union[tuple[int, Optional[int]], list[tuple[int, Optional[int]]]] = (0, None)


class ContrastiveModelConfig(BaseModel):
    backbone: ModelConfig
    projector: ModelConfig
    linear_evaluation_head: HeadConfig


class MultiHeadModelConfig(ModelConfig):
    backbone: ModelConfig
    neck: Optional[ModelConfig] = None
    heads: dict[str, HeadConfig]
