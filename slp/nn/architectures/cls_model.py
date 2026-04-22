from torch import nn, Tensor


class ClassificationModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        out = self.backbone(x, mask)
        out = self.head(out)
        return out
