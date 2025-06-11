from slp.config.templates.module import ModuleConfig
from slp.nn.backbones.mstcn import MSTCNBackbone
from slp.nn.backbones.vit import ViTBackbone


def load_backbone(config: ModuleConfig):
    name, kwargs = config.module_name, config.module_kwargs
    if name == 'mstcn':
        return MSTCNBackbone(**kwargs)
    elif name == 'vit':
        return ViTBackbone(**kwargs)
    else:
        raise ValueError(f'Backbone {name} not supported.')
