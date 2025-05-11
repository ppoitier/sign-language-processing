from slp.config.templates.module import ModuleConfig
from slp.nn.backbones.mstcn import MSTCNBackbone


def load_segmentation_backbone(config: ModuleConfig):
    name, kwargs = config.module_name, config.module_kwargs
    if name == 'mstcn':
        return MSTCNBackbone(**kwargs)
    else:
        raise ValueError(f'Backbone {name} not supported.')
