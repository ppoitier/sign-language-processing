import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from einops import repeat

from slp.nn.blocks.positional_encoding.original import PositionalEncoding


class InputEmbedding(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        max_length: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(c_in, c_out)
        self.positional_encoder = PositionalEncoding(c_out, max_length + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, c_out))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Tensor of shape (N, T, C_in)
            mask: Tensor of shape (N, T)

        Returns:
            out: Tensor of shape (N, T+1, C_out)
            mask: Tensor of shape (N, T+1)
        """
        x = self.input_projection(x)
        cls_tokens = repeat(self.cls_token, "() l c -> b l c", b=x.size(0))
        x = torch.concat((cls_tokens, x), dim=1)
        x = self.positional_encoder(x)
        return self.dropout(x), pad(mask, (1, 0), value=1)


class ViT(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        max_length: int,
        n_heads: int,
        n_layers: int,
        pool: str | None = "cls_token",
    ):
        super().__init__()
        self.input_embedding = InputEmbedding(c_in, max_length, c_out)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_out,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.to_latent = nn.Identity()
        self.pool = pool

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (N, C_in, T) containing the input features.
            mask: Tensor of shape (N, 1, T) where 1's are included and 0's excluded.

        Returns:
            out: Tensor of shape (N, C_out) if pool is 'cls_token' or 'mean', else (N, L+1, C_out)
        """
        # from x: (N, C_in, T), mask: (N, 1, T) to x: (N, T, C_in), mask: (N, T)
        x, mask = x.transpose(1, 2).contiguous(), mask.squeeze(1)
        x, mask = self.input_embedding(x, mask)
        # `src_mask` should be inverted for transformer layers.
        # See https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        x = self.encoder(x, src_key_padding_mask=~(mask.bool()))
        if self.pool is not None:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x
