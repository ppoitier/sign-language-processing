"""
Composable attention pattern factories for ``flex_attention``.

Each factory returns a ``mask_mod`` function compatible with
``create_block_mask``. Combine patterns with ``and_masks``.

Example — padded sliding window::

    from slp.nn.blocks.transformers.attention_patterns import (
        padding_mask_mod,
        sliding_window_mask_mod,
        and_masks,
        build_block_mask,
    )

    mask_mod = and_masks(
        padding_mask_mod(padding),
        sliding_window_mask_mod(128),
    )
    block_mask = build_block_mask(mask_mod, B=N, H=n_heads, Q_LEN=T, KV_LEN=T)
"""

from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask


def padding_mask_mod(padding_mask: Tensor):
    """
    Mask padded key/value positions.

    Args:
        padding_mask: (N, T) bool — ``True`` = valid, ``False`` = pad.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        return padding_mask[b, kv_idx]
    return mask_mod


def sliding_window_mask_mod(window_size: int):
    """Restrict attention to ``window_size`` positions around each query."""
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= window_size
    return mask_mod


def causal_mask_mod():
    """Standard left-to-right causal mask."""
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return mask_mod


def and_masks(*mask_mods):
    """Compose multiple ``mask_mod`` functions with logical AND."""
    def combined(b, h, q_idx, kv_idx):
        result = mask_mods[0](b, h, q_idx, kv_idx)
        for m in mask_mods[1:]:
            result = result & m(b, h, q_idx, kv_idx)
        return result
    return combined


def build_block_mask(mask_mod, B: int, H: int, Q_LEN: int, KV_LEN: int, device):
    """
    Convenience wrapper around ``create_block_mask``.

    Args:
        mask_mod: composed mask function.
        B:        batch size.
        H:        number of attention heads.
        Q_LEN:    query sequence length.
        KV_LEN:   key/value sequence length.
    """
    return create_block_mask(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)