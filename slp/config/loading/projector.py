from slp.config.templates.module import ModuleConfig
from slp.nn.projectors.mlp_block import ProjectionHead


def load_projector(config: ModuleConfig):
    name, kwargs = config.module_name, config.module_kwargs
    if name == 'mlp-block':
        return ProjectionHead(**kwargs)
    else:
        raise ValueError(f'Projector {name} not supported.')