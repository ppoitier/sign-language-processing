from torch import nn

from slp.config.templates.model import MultiHeadModelConfig, HeadConfig, ModelConfig
from slp.nn.blocks.tcn.tcn import MultiStageTCN
from slp.nn.spoter import SPOTER
from slp.nn.model_with_heads import MultiHeadModel, Head


def load_backbone(config: ModelConfig):
    match config.type:
        case 'mstcn':
            return MultiStageTCN(**config.args)
        case 'spoter':
            return SPOTER(**config.args)
        case _:
            raise ValueError(f"Unknown backbone: {config.type}")


def load_head(config: HeadConfig):
    match config.type:
        case 'identity':
            return Head(
                model=nn.Identity(),
                in_channels_range=config.in_channels_range,
            )
        case _:
            raise ValueError(f"Unknown head: {config.type}")


def load_model_architecture(config: MultiHeadModelConfig):
    match config.type:
        case 'multi-head':
            return MultiHeadModel(
                backbone=load_backbone(config.backbone),
                heads=nn.ModuleDict({k: load_head(v) for k, v in config.heads.items()}),
                neck=None,  # TODO: implement this one day
                **config.args,
            )
        case _:
            raise ValueError(f"Unknown model architecture: {config.type}")
