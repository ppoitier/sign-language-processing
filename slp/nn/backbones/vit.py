from torch import nn

from slp.nn.blocks.transformers.vit import ViT


class ViTBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        max_length: int,
        out_channels: int,
        n_heads: int,
        n_layers: int,
        pool: str | None = "cls_token",
    ):
        super().__init__()
        self.backbone = ViT(
            in_channels=in_channels,
            max_length=max_length,
            out_channels=out_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            pool=pool,
        )

    def forward(self, x, mask):
        """
        Args:
            x: tensor of shape (N, T, C_in)
            mask: tensor of shape (N, T)

        Returns:
            logits: tensor of shape (N, T, C_in)
        """
        return self.backbone(x, mask)
