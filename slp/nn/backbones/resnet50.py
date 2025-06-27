from torch import nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.backbone = resnet50()
        self.backbone.fc = nn.Linear(2048, out_features)

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    import torch
    from torchinfo import summary
    B, C_in, H, W = 4, 3, 16, 16
    _x = torch.randn(B, C_in, H, W)
    _model = ResNet50(out_features=500)
    summary(_model, input_data=_x)
