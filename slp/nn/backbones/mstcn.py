from torch import nn

from slp.nn.blocks.tcn import MultiStageTCN


class MSTCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_stages: int = 4,
        n_layers: int = 10,
    ):
        super().__init__()
        self.backbone = MultiStageTCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_stages=n_stages,
            n_layers=n_layers,
        )

    def forward(self, x, mask):
        """
        Args:
            x: tensor of shape (N, T, C_in)
            mask: tensor of shape (N, T)

        Returns:
            logits: tensor of shape (N_layers, N, T, C_in)
        """
        out = self.backbone(
            x.transpose(-1, -2).contiguous(),
            mask.unsqueeze(1).contiguous(),
        )
        return out.transpose(-1, -2).contiguous()
