from torch import nn, Tensor
from torchvision.models import resnet50, ResNet50_Weights


class ResNet_2d(nn.Module):
    def __init__(self, c_out: int = 500, dropout: float = 0.2):
        super().__init__()
        self.r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.r50.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(2048, c_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.r50(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
