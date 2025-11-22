from torch import nn, Tensor
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
    resnet152,
    ResNet152_Weights,
)


class ResNet_2d(nn.Module):
    def __init__(
            self,
            c_out: int = 500,
            dropout: float = 0.2,
            n_layers: int = 50,
            pretrained: bool = True,
    ):
        super().__init__()
        if n_layers == 18:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif n_layers == 50:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        elif n_layers == 152:
            self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            raise ValueError(f"Invalid number of layers: {n_layers}. Must be 18, 50 or 152.")
        self.resnet.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(2048, c_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
