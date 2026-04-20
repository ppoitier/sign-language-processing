"""
Transformer blocks for spatio-temporal data.

Follows the framework convention:
    - Boundary:  (N, C, T), mask (N, 1, T)
    - Internal:  (N, T, C)

The TransformerBlock wraps TransformerEncoderLayer and adds optional
strided downsampling for multi-scale architectures.
"""

from torch import nn, Tensor

from slp.nn.blocks.transformers.layers import TransformerEncoderLayer
from slp.nn.blocks.transformers.attention_patterns import (
    padding_mask_mod,
    and_masks,
    build_block_mask,
)


class TransformerBlock(nn.Module):
    """
    Framework-convention transformer block with optional downsampling.

    Composes a :class:`TransformerEncoderLayer` (pre-norm self-attention +
    FFN using ``flex_attention``) with an optional strided average-pool for
    multi-scale architectures.

    Boundary shapes follow convention: ``(N, C, T)`` in, ``(N, C, T')`` out,
    where ``T' = T // stride``.

    Supports RoPE and composable attention patterns (sliding window,
    causal, etc.) by forwarding them to the underlying layer.

    Args:
        in_channels:     input / model dimension.
        hidden_channels: FFN hidden dimension.
        n_heads:         number of attention heads.
        dropout:         dropout for attention and FFN.
        stride:          temporal downsampling factor (1 = identity).
        rope:            optional rotary embedding module (torchtune-compatible).
        attn_mask_mod:   optional mask mod composed with padding via
                         :func:`and_masks` (e.g., sliding window, causal).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        stride: int = 1,
        rope: nn.Module | None = None,
        attn_mask_mod=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.attn_mask_mod = attn_mask_mod

        self.layer = TransformerEncoderLayer(
            d_model=in_channels,
            n_heads=n_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            rope=rope,
        )

        self.downsample = (
            nn.AvgPool1d(kernel_size=stride, stride=stride) if stride > 1 else None
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: (N, C, T)
            mask: (N, 1, T) — 1 = valid, 0 = pad.
        Returns:
            (N, C, T') where T' = T // stride.
        """
        N, C, T = x.shape

        # -- Build flex_attention BlockMask from padding --
        padding = mask.squeeze(1).bool()
        mask_mod = padding_mask_mod(padding)
        if self.attn_mask_mod is not None:
            mask_mod = and_masks(mask_mod, self.attn_mask_mod)
        block_mask = build_block_mask(
            mask_mod,
            B=N,
            H=self.n_heads,
            Q_LEN=T,
            KV_LEN=T,
            device=mask.device,
        )

        # -- (N, C, T) → (N, T, C) --
        x = x.transpose(1, 2)

        x = self.layer(x, block_mask=block_mask)

        # -- (N, T, C) → (N, C, T) --
        x = x.transpose(1, 2)

        # -- Downsample & re-mask --
        if self.downsample is not None:
            x = self.downsample(x * mask)
            mask = mask[:, :, :: self.downsample.stride[0]]
            x = x * mask

        return x
