from typing import OrderedDict

from torch import nn

from slp.core.registry import BACKBONE_REGISTRY
from slp.core.config.model import ModelConfig
from slp.nn.model_builder import build_backbone


@BACKBONE_REGISTRY.register("chain")
class Chain(nn.Module):
    def __init__(self, backbones: OrderedDict[str, dict]):
        super().__init__()
        self.backbones = nn.ModuleList([
            build_backbone(ModelConfig(name=name, kwargs=kwargs)) for name, kwargs in backbones.items()
        ])

    def forward(self, x, mask=None):
        for b in self.backbones:
            x = b(x, mask)
        return x
