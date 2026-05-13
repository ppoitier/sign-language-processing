from typing import Optional

from torch import nn, Tensor
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
)

from slp.core.registry import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("resnet2d")
class ResNet2D(nn.Module):
    """
    A 2D ResNet backbone operating on landmark-time feature maps.

    Shape Notation:
        - N: Batch size.
        - C_in: Number of input channels.
        - C_out: Number of output channels.
        - L: Number of landmarks (image height).
        - T: Temporal sequence length (image width).
    """

    _CONFIGS: dict = {
        18:  (resnet18,  ResNet18_Weights.IMAGENET1K_V1,  512),
        34:  (resnet34,  ResNet34_Weights.IMAGENET1K_V1,  512),
        50:  (resnet50,  ResNet50_Weights.IMAGENET1K_V2,  2048),
        101: (resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
        152: (resnet152, ResNet152_Weights.IMAGENET1K_V2, 2048),
    }

    def __init__(
        self,
        num_layers: int = 50,
        pretrained: bool = True,
    ):
        super().__init__()

        if num_layers not in self._CONFIGS:
            raise ValueError(
                f"num_layers must be one of {list(self._CONFIGS.keys())}, "
                f"got {num_layers}"
            )

        factory, weights, embed_dim = self._CONFIGS[num_layers]
        self.c_out = embed_dim
        self.resnet = factory(weights=weights if pretrained else None)
        self.resnet.fc = nn.Identity()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:    Input tensor of shape (N, C_in, L, T).
            mask: Unused. Accepted for API compatibility.

        Returns:
            Output tensor of shape (N, C_out).
        """
        return self.resnet(x)