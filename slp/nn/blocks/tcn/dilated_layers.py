from torch import nn


class DilatedResidualLayer(nn.Module):
    def __init__(self, in_channels: int, dilation: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, in_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_out = nn.Conv1d(in_channels, in_channels, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        """
        Args:
            x: input tensor of shape (N, C_in, T)
            mask: mask tensor of shape (N, 1, T)

        Returns:
            output tensor of shape (N, C_in, T)
        """
        # x: (N, C_in, L)
        out = self.act(self.conv_dilated(x))
        out = self.conv_out(out)
        return (self.dropout(out) + x) * mask
