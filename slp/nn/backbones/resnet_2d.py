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
        n_layers: int = 50,
        pretrained: bool = True,
    ):
        super().__init__()

        if n_layers not in self._CONFIGS:
            raise ValueError(
                f"num_layers must be one of {list(self._CONFIGS.keys())}, "
                f"got {n_layers}"
            )

        factory, weights, embed_dim = self._CONFIGS[n_layers]
        self.c_out = embed_dim
        self.resnet = factory(weights=weights if pretrained else None)
        self.resnet.fc = nn.Identity()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:    Input tensor of shape (N, C_in, H, W) or (N, T, C_in, H, W).
            mask: Unused. Accepted for API compatibility.

        Returns:
            Output tensor of shape (N, C_out) or (N, C_out, T).
        """
        batch_size = x.size(0)
        collapsed = x.dim() == 5
        if collapsed:
            x = x.flatten(0, 1)  # (N, T, C, H, W) -> (N*T, C, H, W)
        out = self.resnet(x)
        if collapsed:
            out = out.reshape(batch_size, -1, out.size(-1))  # (N*T, C_out) -> (N, T, C_out)
            out = out.transpose(1, 2)  # (N, T, C_out) -> (N, C_out, T)
        return out


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 3, 65, 64)
    model = ResNet2D()
    y = model(x)
    print(y.size())
