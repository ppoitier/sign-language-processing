# noinspection PyUnusedImports
import slp.nn.backbones
# noinspection PyUnusedImports
import slp.nn.necks
# noinspection PyUnusedImports
import slp.nn.heads

from torch import nn
from slp.nn.architectures.hydra import HydraModel
from slp.nn.heads.multi_stage import SharedMultiStageHead, SingleStageHead
from slp.nn.heads.channel_splitter import TaskChannelSplitter
from slp.core.config.model import HydraConfig, HeadConfig
from slp.core.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, NECK_REGISTRY


def load_head(head_config: HeadConfig) -> nn.Module:
    """Instantiates a single base head from its configuration."""
    head_cls = HEAD_REGISTRY.get(head_config.name)
    return head_cls(**head_config.kwargs)


def build_hydra_model(config: HydraConfig) -> HydraModel:
    """Builds the complete multi-task architecture."""

    # 1. Build Backbone
    backbone_cls = BACKBONE_REGISTRY.get(config.backbone.name)
    backbone = backbone_cls(**config.backbone.kwargs)

    # 2. Build Neck (Optional)
    neck = None
    if config.neck is not None:
        neck_cls = NECK_REGISTRY.get(config.neck.name)
        neck = neck_cls(**config.neck.kwargs)

    # 3. Build Base Heads
    heads = {
        name: load_head(head_config)
        for name, head_config in config.heads.items()
    }

    # 4. Build the Channel Splitter
    split_sections = [head_config.n_channels for head_config in config.heads.values()]
    multi_task_splitter = TaskChannelSplitter(
        split_sections=split_sections,
        heads=heads
    )

    # 5. Wrap in Multi-Stage logic
    if config.multi_layer:
        head_assembly = SharedMultiStageHead(multi_task_splitter)
    else:
        head_assembly = SingleStageHead(multi_task_splitter)

    # 6. Assemble the HydraModel
    return HydraModel(
        backbone=backbone,
        neck=neck,
        head_assembly=head_assembly
    )