from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# from slp.nn.blocks.transformers.vit import ViT


class ViT(nn.Module):
    def __init__(
        self,
        c_out: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()

        self.pad = nn.ConstantPad2d(padding=(0, 224 - 64, 0, 224 - 65), value=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(768, c_out)

    def forward(self, x, mask):
        """
        Args:
            x: tensor of shape (N, C_in, T)
            mask: tensor of shape (N, 1, T)
        """
        x = self.pad(x)
        x = self.vit(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    import torch

    model = ViT(c_out=500)
    x = torch.randn(1, 3, 65, 64)
    y = model(x, None)
    print(model)
    print(y.shape)
