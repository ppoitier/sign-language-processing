"""
Transformer backbones for spatio-temporal data.

All backbones follow the framework convention:
    - Input:  x (N, C, T), mask (N, 1, T) where 1=valid, 0=pad
    - Output: list[Tensor] (one per stage) or Tensor

The (N, C, T) <--> (N, T, C) transpose happens at each backbone's
boundary. Core layers operate in standard PyTorch (N, T, C).

Architectures:
    MultiStageTransformer    — multi-scale dense features via staged
                               downsampling
    TransformerViT           — encoder-only, flexible pooling
    TransformerQueryReadout  — cross-attention classification

Requires:
    - PyTorch >= 2.5   (flex_attention)
    - torchtune         (RotaryPositionalEmbeddings)
"""

import torch
from torch import nn, Tensor
from torchtune.modules import RotaryPositionalEmbeddings

from slp.core.registry import BACKBONE_REGISTRY
from slp.nn.blocks.positional_encoding.sinusoidal import PositionalEncoding
from slp.nn.blocks.transformers.blocks import TransformerBlock
from slp.nn.blocks.transformers.layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from slp.nn.blocks.transformers.attention_patterns import (
    padding_mask_mod,
    and_masks,
    build_block_mask,
    get_attn_mask_mod,
)
from slp.nn.backbones.multi_stage import MultiStageBackbone


@BACKBONE_REGISTRY.register("ms-transformer")
class MultiStageTransformer(nn.Module):
    """
    Hierarchical transformer for dense temporal prediction.

    Projects the input to ``hidden_channels``, applies positional
    encoding, then passes through two kinds of stages:

        proj_in -> PE -> Stage 0 (stride-1 layers)
                      -> Stage 1..N (stride-2 layers via MultiStageBackbone)

    Each layer is a :class:`TransformerBlock` backed by
    ``flex_attention``, with optional RoPE and composable attention
    patterns.

    Args:
        in_channels:      input feature dimension.
        hidden_channels:  model dimension after projection.
        max_length:       maximum sequence length for positional encoding.
        n_heads:          number of attention heads.
        dim_feedforward:  FFN hidden dimension inside each layer.
        n_stem_layers:  number of full resolution layers.
        n_branch_layers:   number of stride-2 layers (one per downsampling
                          stage in MultiStageBackbone).
        dropout:          dropout for attention and FFN.
        pos_encoding:     ``"rope"`` | ``"sinusoidal"`` | ``None``.
        attn_mask_mod:    optional mask mod (e.g. sliding window) composed
                          with padding in every layer.

    Returns:
        list[Tensor] of shape (N, hidden_channels, T/2^i) per stage.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_length: int,
        n_heads: int = 4,
        dim_feedforward: int = 2048,
        n_stem_layers: int = 2,
        n_branch_layers: int = 5,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        attn_mask_strategy: str | None = None,
    ):
        super().__init__()

        self.proj_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        attn_mask_mod = get_attn_mask_mod(attn_mask_strategy)

        # -- Positional encoding --
        head_dim = hidden_channels // n_heads
        rope = None
        self.additive_pe = None
        if pos_encoding == "rope":
            rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_length)
        elif pos_encoding == "sinusoidal":
            self.additive_pe = PositionalEncoding(hidden_channels, max_length)

        # -- Stage 0: full resolution --
        self.stage0 = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_channels,
                    dim_feedforward,
                    n_heads=n_heads,
                    dropout=dropout,
                    rope=rope,
                    attn_mask_mod=attn_mask_mod,
                )
                for _ in range(n_stem_layers)
            ]
        )

        # -- Stage 1..N: stride-2 downsampling --
        self.stages = MultiStageBackbone(
            [
                TransformerBlock(
                    hidden_channels,
                    dim_feedforward,
                    n_heads=n_heads,
                    stride=2,
                    dropout=dropout,
                    rope=rope,
                    attn_mask_mod=attn_mask_mod,
                )
                for _ in range(n_branch_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]:
        """
        Args:
            x:    (N, C_in, T) -- framework convention.
            mask: (N, 1, T)    -- 1=valid, 0=pad.
        Returns:
            list[Tensor] of (N, hidden_channels, T/2^i), one per stage.
        """
        x = self.proj_in(x)

        if self.additive_pe is not None:
            x = self.additive_pe(x)

        for layer in self.stage0:
            x = layer(x, mask)

        return self.stages(x, mask)


@BACKBONE_REGISTRY.register("vit")
class TransformerViT(nn.Module):
    """
    ViT-style encoder-only transformer.

    Output modes via ``pool``:
        - ``None``   -> (N, C, T)  dense features
        - ``"mean"`` -> (N, C)     mean-pooled
        - ``"cls"``  -> (N, C)     CLS token readout

    Attention pattern is composable: pass ``attn_mask_mod`` to add
    sliding window, causal masking, etc. on top of padding.

    Example::

        from slp.nn.blocks.transformers.attention_patterns import sliding_window_mask_mod

        backbone = TransformerViT(
            ...,
            attn_mask_mod=sliding_window_mask_mod(128),
        )
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_length: int,
        n_heads: int = 4,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        pool: str | None = None,
        attn_mask_strategy: str | None = None,
    ):
        super().__init__()
        self.pool = pool
        self.n_heads = n_heads
        self.attn_mask_mod = get_attn_mask_mod(attn_mask_strategy)

        head_dim = hidden_channels // n_heads
        pe_length = max_length + 1 if pool == "cls" else max_length

        self.proj_in = nn.Linear(in_channels, hidden_channels)

        # -- Positional encoding --
        rope = None
        self.additive_pe = None
        if pos_encoding == "rope":
            rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=pe_length)
        elif pos_encoding == "sinusoidal":
            self.additive_pe = PositionalEncoding(hidden_channels, pe_length)

        # -- CLS token --
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        # -- Encoder layers --
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_channels,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope=rope,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    (N, C_in, T) -- framework convention.
            mask: (N, 1, T)    -- 1=valid, 0=pad.
        Returns:
            (N, hidden_channels, T) if pool is None, else (N, hidden_channels).
        """
        N = x.size(0)

        # -- (N, C, T) -> (N, T, C) --
        x = x.transpose(1, 2)
        padding = mask.squeeze(1).bool()

        x = self.proj_in(x)

        if self.pool == "cls":
            cls = self.cls_token.expand(N, -1, -1)
            x = torch.cat([cls, x], dim=1)
            padding = torch.cat(
                [
                    torch.ones(N, 1, device=padding.device, dtype=torch.bool),
                    padding,
                ],
                dim=1,
            )

        if self.additive_pe is not None:
            x = self.additive_pe(x.transpose(1, 2)).transpose(1, 2)

        # -- Build attention mask --
        T = x.size(1)
        mask_mod = padding_mask_mod(padding)
        if self.attn_mask_mod is not None:
            mask_mod = and_masks(mask_mod, self.attn_mask_mod)
        block_mask = build_block_mask(
            mask_mod, B=N, H=self.n_heads, Q_LEN=T, KV_LEN=T, device=x.device
        )

        for layer in self.layers:
            x = layer(x, block_mask=block_mask)

        # -- Pool & return --
        if self.pool == "cls":
            return x[:, 0]
        elif self.pool == "mean":
            lengths = padding.sum(dim=1, keepdim=True).unsqueeze(-1)
            return (x * padding.unsqueeze(-1)).sum(dim=1) / lengths.squeeze(-1).clamp(
                min=1
            )
        else:
            return (x.transpose(1, 2)) * mask


@BACKBONE_REGISTRY.register("transformer-query-readout")
class TransformerQueryReadout(nn.Module):
    """
    Encoder with cross-attention query readout for classification.

    A learned query probes the encoded memory via cross-attention
    (DETR-style). Set ``n_queries > 1`` for multi-label or detection.

    Returns:
        (N, hidden_channels) for n_queries=1,
        (N, n_queries, hidden_channels) otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        max_length: int,
        n_heads: int = 4,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        n_queries: int = 1,
        attn_mask_strategy: str | None = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.attn_mask_mod = get_attn_mask_mod(attn_mask_strategy)

        head_dim = hidden_channels // n_heads

        self.proj_in = nn.Linear(in_channels, hidden_channels)

        # -- Positional encoding --
        rope = None
        self.additive_pe = None
        if pos_encoding == "rope":
            rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_length)
        elif pos_encoding == "sinusoidal":
            self.additive_pe = PositionalEncoding(hidden_channels, max_length)

        # -- Encoder --
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_channels,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope=rope,
                )
                for _ in range(n_encoder_layers)
            ]
        )

        # -- Decoder --
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=hidden_channels,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope=rope,
                )
                for _ in range(n_decoder_layers)
            ]
        )

        self.class_query = nn.Parameter(torch.randn(1, n_queries, hidden_channels))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    (N, C_in, T) -- framework convention.
            mask: (N, 1, T)    -- 1=valid, 0=pad.
        Returns:
            (N, hidden_channels) if n_queries=1,
            (N, n_queries, hidden_channels) otherwise.
        """
        N = x.size(0)

        # -- (N, C, T) -> (N, T, C) --
        x = x.transpose(1, 2)
        padding = mask.squeeze(1).bool()

        x = self.proj_in(x)

        if self.additive_pe is not None:
            x = self.additive_pe(x.transpose(1, 2)).transpose(1, 2)

        # -- Encode --
        T = x.size(1)
        mask_mod = padding_mask_mod(padding)
        if self.attn_mask_mod is not None:
            mask_mod = and_masks(mask_mod, self.attn_mask_mod)
        enc_block_mask = build_block_mask(
            mask_mod,
            B=N,
            H=self.n_heads,
            Q_LEN=T,
            KV_LEN=T,
            device=x.device,
        )

        for layer in self.encoder_layers:
            x = layer(x, block_mask=enc_block_mask)

        memory = x

        # -- Decode --
        query = self.class_query.expand(N, -1, -1)
        cross_mask_mod = padding_mask_mod(padding)
        cross_block_mask = build_block_mask(
            cross_mask_mod,
            B=N,
            H=self.n_heads,
            Q_LEN=self.n_queries,
            KV_LEN=T,
            device=x.device,
        )

        for layer in self.decoder_layers:
            query = layer(query, memory, cross_block_mask=cross_block_mask)

        if self.n_queries == 1:
            return query.squeeze(1)
        return query
