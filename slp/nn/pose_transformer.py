import torch
from torch import nn, Tensor, BoolTensor

from slp.nn.blocks.positional_encoding.original import PositionalEncoding


class PoseTransformer(nn.Module):
    def __init__(
            self,
            c_in: int = 130,
            c_hidden: int = 128,
            c_out: int = 500,
            max_length: int = 64,
            n_heads: int = 4,
            n_encoder_layers: int = 6,
            n_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.fc_in = nn.Linear(c_in, c_hidden)
        self.pe = PositionalEncoding(in_channels=c_hidden, max_length=max_length)
        self.class_query = nn.Parameter(torch.rand(1, 1, c_hidden))
        self.transformer = nn.Transformer(
            d_model=c_hidden,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.fc_out = nn.Linear(c_hidden, c_out)

    def forward(self, x: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x: tensor of shape (N, C_in, T)
            mask: bool tensor of shape (N, 1, T) where:
                - true: use the element of the sequence
                - false: don't use the element

        Returns:
            # logits: tensor of shape (N, C_out)
        """
        # from (N, C_in, T) to (N, T, C_in)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc_in(x)
        x = self.pe(x)

        # Repeat the class query for each item in the batch.
        batch_size = x.shape[0]
        query_embed = self.class_query.repeat(batch_size, 1, 1)

        # output of shape (N, 1, C_hidden)
        padding_mask = ~mask[:, 0]  # Shape (N, T)
        out = self.transformer(
            x,
            query_embed,
            src_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        # from (N, 1, C_hidden) to (N, C_out)
        out = self.fc_out(out).squeeze(1)
        return out
