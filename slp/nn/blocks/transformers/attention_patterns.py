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
import torch.compiler
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
    """Restrict attention to ``window_size`` centered on the query."""
    half = window_size // 2
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= half
    return mask_mod


def causal_mask_mod():
    """Standard left-to-right causal mask."""
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return mask_mod


def global_tokens_mask_mod(n_global: int):
    """
    Allow the first ``n_global`` tokens to attend to everything,
    and everything to attend to them. Useful as 'attention sinks'
    or Longformer-style global tokens, typically combined with
    a sliding window.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx < n_global) | (kv_idx < n_global)
    return mask_mod


def and_masks(*mask_mods):
    """Compose multiple ``mask_mod`` functions with logical AND."""
    def combined(b, h, q_idx, kv_idx):
        result = mask_mods[0](b, h, q_idx, kv_idx)
        for m in mask_mods[1:]:
            result = result & m(b, h, q_idx, kv_idx)
        return result
    return combined


@torch.compiler.disable
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


_PATTERN_REGISTRY = {
    "causal": lambda: causal_mask_mod(),
    "window": lambda size: sliding_window_mask_mod(int(size)),
    "sink":   lambda n:    global_tokens_mask_mod(int(n)),
    "global": lambda n:    global_tokens_mask_mod(int(n)),
}


def get_attn_mask_mod(spec: str | None):
    """
    Build a composed mask_mod from a string spec.

    Grammar:  <pattern>[:<arg>] ( '+' <pattern>[:<arg>] )*

    Examples:
        None                  -> None (dense attention)
        "causal"              -> causal
        "window:128"          -> sliding window of size 128
        "window:128+causal"   -> causal sliding window
        "window:256+sink:4"   -> windowed + 4 attention-sink tokens

    Padding is *not* included here — it's added per-batch inside the
    backbone (it depends on the actual mask tensor).
    """
    if spec is None or spec == "" or spec == "none":
        return None

    mods = []
    for part in spec.split("+"):
        name, _, arg = part.strip().partition(":")
        if name not in _PATTERN_REGISTRY:
            raise ValueError(
                f"Unknown attention pattern '{name}'. "
                f"Available: {list(_PATTERN_REGISTRY)}"
            )
        factory = _PATTERN_REGISTRY[name]
        mods.append(factory(arg) if arg else factory())

    if len(mods) == 1:
        return mods[0]
    return and_masks(*mods)
