# noinspection PyUnusedImports
import slp.nn.backbones

# noinspection PyUnusedImports
import slp.nn.heads
from slp.nn.heads.multi_stage import SharedMultiStageHead
from slp.nn.heads.channel_splitter import TaskChannelSplitter
from slp.core.config.model import HydraConfig, HeadConfig
from slp.core.registry import BACKBONE_REGISTRY, HEAD_REGISTRY


def load_head(model_config: HydraConfig, head_config: HeadConfig): ...


def build_hydra_model(config: HydraConfig):
    backbone_cls = BACKBONE_REGISTRY.get(config.backbone.name)
    backbone = backbone_cls(**config.backbone.kwargs)

    print(backbone)

    heads = {
        k: HEAD_REGISTRY.get(head_config.name)(**head_config.kwargs)
        for k, head_config in config.heads.items()
    }
    print(heads)
