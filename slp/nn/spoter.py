import copy
from typing import Optional

import torch
from torch import nn, Tensor


def _get_clones(mod, n):
    """Helper function to duplicate a module N times."""
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention
    operation and using the author's specific forward pass for reproducibility.
    @see https://github.com/maty-bohacek/spoter/blob/main/spoter/spoter_model.py
    """

    def __init__(self, **kwargs):
        super(SPOTERTransformerDecoderLayer, self).__init__(**kwargs)
        # The key architectural change: remove self-attention
        # del self.self_attn
        # CHANGES: We do run delete it because it is needed in the forward function to get the sequence length.

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:

        # In the original code, dropout and norm are applied before the multi-head attention.
        # This is unusual, as they typically follow an attention block. We keep it for fidelity.
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        # Cross-Attention step (the only attention mechanism left in this layer)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-Forward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Faithful implementation of the SPOTER (Sign POse-based TransformER) architecture.
    @see https://github.com/maty-bohacek/spoter/blob/main/spoter/spoter_model.py
    """

    def __init__(
        self,
        n_classes: int,
        c_in: int = 108,
        max_sequence_lengths=200,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        assert c_in % n_heads == 0, "The input dimension must be divisible by the number of heads."
        # -- Learnable parameters
        self.pos_embed = nn.Parameter(torch.rand(1, max_sequence_lengths, c_in))
        # The single "Class Query" vector for the decoder
        self.class_query = nn.Parameter(torch.rand(1, 1, c_in))

        self.transformer = nn.Transformer(
            d_model=c_in,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # -- Modules
        # Create a custom decoder layer and overwrite the default layers inside the already-built transformer.
        # custom_decoder_layer = SPOTERTransformerDecoderLayer(
        #     d_model=self.transformer.d_model,
        #     nhead=self.transformer.nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation="relu",
        #     batch_first=True,
        # )
        # self.transformer.decoder.layers = _get_clones(
        #     custom_decoder_layer, n_decoder_layers
        # )
        self.linear_class = nn.Linear(c_in, n_classes)

    def forward(
        self,
            x: torch.Tensor,
            masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input sequence of pose vectors. Tensor of shape (N, T, C_in).
            masks (Tensor): Boolean mask tensor of shape (N, T), where 1=included, and 0=excluded.

        Returns:
            output: Tensor of shape (N, n_classes)
        """
        pos_encoded_src = x + self.pos_embed[:, : x.shape[1], :]

        # Repeat the class query for each item in the batch.
        batch_size = x.shape[0]
        query_embed = self.class_query.repeat(batch_size, 1, 1)

        # Pass data through the transformer. Output of shape (B, 1, C_h) as we only have 1 query token.
        h = self.transformer(
            pos_encoded_src, query_embed, src_key_padding_mask=~masks
        )
        # Pass through the final classifier and remove the sequence dimension
        res = self.linear_class(h).squeeze(1)

        return res


if __name__ == "__main__":
    # Example usage:
    model = SPOTER(n_classes=20, c_in=130, n_heads=10)
    # (B, S, D) -> Batch of 4, 50 frames, 108 features per frame
    video_frames = torch.rand(4, 50, 130)
    output = model(video_frames)
    print(output.shape)  # Should be torch.Size([4, 20])
