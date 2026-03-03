# noinspection PyUnusedImports
import slp.nn.backbones
from slp.core.registry import BACKBONE_REGISTRY


if __name__ == '__main__':
    model_class = BACKBONE_REGISTRY.get("ms-tcn")
    print(model_class)
