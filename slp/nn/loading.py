from torch import nn

from slp.config.templates.model import MultiHeadModelConfig, HeadConfig, ModelConfig
from slp.nn.blocks.tcn.tcn import MultiStageTCN
from slp.nn.blocks.i3d.original import InceptionI3d
from slp.nn.model_with_heads import MultiHeadModel, Head
from slp.nn.projectors.mlp_block import ProjectionHead
from slp.nn.pose_transformer import PoseTransformer
from slp.nn.backbones.resnet_3d import ResNet_3d
from slp.nn.backbones.resnet_2d import ResNet_2d
from slp.nn.blocks.transformers.vit import PoseViT


def load_backbone(config: ModelConfig):
    match config.type:
        case 'mstcn':
            return MultiStageTCN(**config.args)
        case 'spoter':
            return PoseTransformer(**config.args)
        case 'pose-vit':
            return PoseViT(**config.args)
        case 'inception-i3d':
            return InceptionI3d(**config.args)
        case 'resnet50-i3d':
            return ResNet_3d(**config.args)
        case 'resnet50-2d':
            return ResNet_2d(**config.args)
        case _:
            raise ValueError(f"Unknown backbone: {config.type}")


def load_projector(config: ModelConfig):
    match config.type:
        case 'mlp':
            return ProjectionHead(**config.args)
        case _:
            raise ValueError(f"Unknown projector: {config.type}")


def load_head(config: HeadConfig):
    match config.type:
        case 'identity':
            return Head(
                model=nn.Identity(),
                in_channels_range=config.in_channels_range,
            )
        case 'linear':
            return Head(
                model=nn.Linear(**config.args),
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
