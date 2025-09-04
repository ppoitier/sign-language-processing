from torch import nn, Tensor


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head self-attention block with convolutional projection.
    """
    def __init__(
        self,
        in_features: int,
        n_heads: int,
        attn_drop: float = 0.0,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = in_features // n_heads

        # A single convolutional layer for Q, K, V projection
        self.qkv_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features * 3,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            stride=conv_stride,
            bias=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=n_heads,
            dropout=attn_drop,
            batch_first=True, # Expects (N, L, C)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (N, C, L)
        qkv = self.qkv_conv(x)
        # Split into Q, K, V
        # qkv shape: (N, 3 * C, L')
        q, k, v = qkv.transpose(1, 2).contiguous().chunk(3, dim=2) # each is (N, L', C)
        # Attention mechanism
        attn_output, _ = self.attn(q, k, v, need_weights=False)
        # Transpose back to (N, C, L')
        return attn_output.transpose(1, 2).contiguous()
