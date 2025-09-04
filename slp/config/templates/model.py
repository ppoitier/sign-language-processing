from collections import OrderedDict

from pydantic import BaseModel


class HeadConfig(BaseModel):
    in_channels: int
    out_channels: int
    layer: str = 'linear'


class ModelConfig(BaseModel):
    name: str
    encoder: dict = {}
    heads: OrderedDict[str, HeadConfig] = {}
    checkpoint_path: str | None = None