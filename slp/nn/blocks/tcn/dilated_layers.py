from torch import nn, Tensor


class DilatedResidualLayer(nn.Module):
    """
    A 1D dilated residual layer.

    Shape Notation:
        - N: Batch size.
        - C: Number of channels.
        - T: Temporal sequence length.
    """

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, act: nn.Module | None = None):
        """
        Args:
            channels: Number of input and output channels (invariant for residual).
            dilation: Dilation factor for the 3-kernel 1D convolution.
            kernel_size: Kernel size for the 1D convolution (default to 3).
            act: Activation function for the 1D convolution (default to nn.ReLU()).
        """
        super().__init__()
        # padding = dilation ensures the temporal dimension T remains constant for kernel_size=3
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=dilation, dilation=dilation
        )
        self.conv_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.ReLU() if act is None else act
        self.dropout = nn.Dropout()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            Output tensor of shape (N, C, T) with invalid padded regions zeroed out.
        """
        out = self.act(self.conv_dilated(x))
        out = self.conv_out(out)
        return (self.dropout(out) + x) * mask
