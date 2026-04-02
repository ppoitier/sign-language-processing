"""
Transformer layers built on ``flex_attention``.

All tensors use standard PyTorch convention: (N, T, C).
Positional encoding is injected via an optional ``rope`` module
(e.g. ``torchtune.modules.RotaryPositionalEmbeddings``) applied
to Q and K inside every layer.

Classes:
    TransformerEncoderLayer — self-attention + FFN.
    TransformerDecoderLayer — self-attention + cross-attention + FFN.
"""

from torch import nn, Tensor
from torch.nn.attention.flex_attention import flex_attention


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm encoder layer: LayerNorm → self-attention → residual → FFN.

    Args:
        d_model:         model dimension.
        n_heads:         number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout:         dropout rate.
        rope:            optional rotary embedding module (torchtune-compatible).
                         Must accept (N, T, H, D) tensors.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        rope: nn.Module | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.rope = rope

    def forward(self, x: Tensor, block_mask=None) -> Tensor:
        """
        Args:
            x:          (N, T, C)
            block_mask: BlockMask from ``create_block_mask``.
        """
        N, T, C = x.shape

        # -- Self-Attention --
        residual = x
        x = self.norm1(x)

        qkv = self.qkv_proj(x).reshape(N, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (N, T, H, D)

        if self.rope is not None:
            q = self.rope(q)  # torchtune: (N, T, H, D) → (N, T, H, D)
            k = self.rope(k)

        q = q.transpose(1, 2)  # (N, H, T, D) for flex_attention
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = attn_out.transpose(1, 2).reshape(N, T, C)

        x = residual + self.attn_drop(self.out_proj(attn_out))

        # -- FFN --
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Pre-norm decoder layer: self-attention → cross-attention → FFN.

    Cross-attention: Q from ``tgt``, K/V from ``memory``.
    RoPE is applied to memory keys in cross-attention to encode
    source positions (query tokens are learned, not positional).

    Args:
        d_model:         model dimension.
        n_heads:         number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout:         dropout rate.
        rope:            optional rotary embedding for memory keys.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        rope: nn.Module | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Self-attention
        self.self_qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.self_out_proj = nn.Linear(d_model, d_model)

        # Cross-attention
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.cross_out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        cross_block_mask=None,
    ) -> Tensor:
        """
        Args:
            tgt:              (N, T_q, C) — query tokens (often T_q=1).
            memory:           (N, T_kv, C) — encoded source.
            cross_block_mask: BlockMask for cross-attention padding.
        """
        N, T_q, C = tgt.shape
        T_kv = memory.size(1)
        H, D = self.n_heads, self.head_dim

        # -- Self-attention on queries --
        residual = tgt
        x = self.norm1(tgt)
        qkv = self.self_qkv_proj(x).reshape(N, T_q, 3, H, D)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
        attn_out = flex_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(N, T_q, C)
        tgt = residual + self.dropout(self.self_out_proj(attn_out))

        # -- Cross-attention into memory --
        residual = tgt
        x = self.norm2(tgt)
        q = self.cross_q_proj(x).reshape(N, T_q, H, D)
        kv = self.cross_kv_proj(memory).reshape(N, T_kv, 2, H, D)
        k, v = kv.unbind(dim=2)  # each (N, T_kv, H, D)

        if self.rope is not None:
            k = self.rope(k)

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        cross_out = flex_attention(q, k, v, block_mask=cross_block_mask)
        cross_out = cross_out.transpose(1, 2).reshape(N, T_q, C)
        tgt = residual + self.dropout(self.cross_out_proj(cross_out))

        # -- FFN --
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt
