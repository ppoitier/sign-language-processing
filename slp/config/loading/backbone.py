from slp.config.templates.module import ModuleConfig
from slp.nn.backbones.mstcn import MSTCNBackbone
from slp.nn.backbones.vit import ViTBackbone
from slp.nn.backbones.resnet50 import ResNet50


def load_backbone(config: ModuleConfig):
    name, kwargs = config.module_name, config.module_kwargs
    if name == 'mstcn':
        return MSTCNBackbone(**kwargs)
    elif name == 'vit':
        return ViTBackbone(**kwargs)
    elif name == 'resnet50':
        return ResNet50(**kwargs)
    else:
        raise ValueError(f'Backbone {name} not supported.')
