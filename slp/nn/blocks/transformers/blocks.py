from torch import nn, Tensor

from slp.nn.blocks.norm.layer_norm import LayerNorm
from slp.nn.blocks.transformers.stability import AffineDropPath
from slp.nn.blocks.transformers.mha import MultiHeadAttentionBlock


class Mlp(nn.Module):
    """
    A simple MLP block using 1D convolutions.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_heads: int,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        path_drop: float = 0.0,
        conv_stride: int = 1,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.norm1 = LayerNorm(in_channels)
        self.attn = MultiHeadAttentionBlock(
            in_features=in_channels,
            n_heads=n_heads,
            attn_drop=attn_drop,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
        )
        self.path_drop1 = AffineDropPath(in_channels, drop_p=path_drop)

        self.norm2 = LayerNorm(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=hidden_channels,
            drop=mlp_drop,
        )
        self.path_drop2 = AffineDropPath(in_channels, drop_p=path_drop)

        self.skip_pool = (
            nn.MaxPool1d(
                kernel_size=conv_stride,
                stride=conv_stride,
                padding=0,
            )
            if conv_stride > 1
            else nn.Identity()
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        shortcut = x
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = self.skip_pool(shortcut) + self.path_drop1(attn_out)

        # Second sub-layer: MLP
        shortcut = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = shortcut + self.path_drop2(mlp_out)
        return x


if __name__ == "__main__":
    import torch

    N, C, L = 3, 32, 128
    _x = torch.randn(N, C, L)
    _model = TransformerBlock(in_channels=C, hidden_channels=C//2, n_heads=4, conv_stride=2)
    _y = _model(_x)
    print(_y.shape)