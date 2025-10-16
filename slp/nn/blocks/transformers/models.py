from torch import nn, Tensor

from slp.nn.multi_stage import MultiStageModel
from slp.nn.blocks.positional_encoding.original import PositionalEncoding
from slp.nn.blocks.transformers.blocks import TransformerBlock


class MultiStageTransformerModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            max_length: int,
            n_stem_layers: int = 2,
            n_branch_layers: int = 5,
            n_heads: int = 4,
    ):
        super().__init__()

        self.pe = PositionalEncoding(in_channels, max_length)
        self.stem = nn.ModuleList([
            TransformerBlock(in_channels, in_channels, n_heads=n_heads)
            for _ in range(n_stem_layers)
        ])
        self.branches = MultiStageModel([
            TransformerBlock(in_channels, in_channels, n_heads=n_heads, conv_stride=2)
            for _ in range(n_branch_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor):
        x = self.pe(x)
        for layer in self.stem:
            x = layer(x, mask)
        return self.branches(x, mask)


if __name__ == '__main__':
    import torch
    N, C_in, L = 3, 32, 128

    _x = torch.randn(N, C_in, L)
    _mask = torch.ones(N, 1, L)

    _model = MultiStageTransformerModel(
        in_channels=C_in,
        max_length=L,
        n_stem_layers=2,
        n_branch_layers=5,
    )
    _model(_x, _mask)
