import torch
from torch import nn

from slp.nn.blocks.tcn import MultiStageTCN
from slp.nn.heads.trident import TridentHead


class MSTCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_stages: int = 4,
        n_layers: int = 10,
        with_trident_head: bool = False,
        n_trident_bins: int = 16,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.backbone = MultiStageTCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_stages=n_stages,
            n_layers=n_layers,
        )
        self.trident_head = None
        if with_trident_head:
            self.trident_head = TridentHead(n_trident_bins)
            self.n_trident_logits = 2 * (n_trident_bins + 1)
            assert out_channels > self.n_trident_logits, f"The trident head need at least {self.n_trident_logits} output channels to compute the offsets (in addition of score channels)."
            self.n_score_logits = out_channels - self.n_trident_logits

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
        out = out.transpose(-1, -2).contiguous()
        if self.trident_head is not None:
            score_logits, trident_logits = torch.split(out, [self.n_score_logits, self.n_trident_logits], dim=-1)
            out = self.trident_head(trident_logits)
            out = torch.stack([score_logits, out], dim=-1)
        return out
