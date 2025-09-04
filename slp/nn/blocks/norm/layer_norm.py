from torch import nn


class LayerNorm(nn.LayerNorm):
    """
    Layer Normalization that supports inputs of shape (N, C, T)
    by transposing before and after the operation.
    """
    def forward(self, x):
        # Input shape (N, C, T) -> transpose to (N, T, C) for LayerNorm
        output = super().forward(x.transpose(1, 2))
        # Transpose back to (N, C, T)
        return output.transpose(1, 2)
