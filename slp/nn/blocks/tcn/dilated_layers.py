import torch
from torch import nn


class DilatedResidualLayer(nn.Module):
    def __init__(self, in_channels: int, layer_idx: int):
        super().__init__()
        dilation = 2**layer_idx
        self.conv_dilated = nn.Conv1d(
            in_channels, in_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_out = nn.Conv1d(in_channels, in_channels, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        # x: (N, C_in, L)
        out = self.act(self.conv_dilated(x))
        out = self.conv_out(out)
        return (self.dropout(out) + x) * mask


class DualDilatedResidualLayer(nn.Module):
    def __init__(self, in_channels: int, layer_idx: int, n_layers: int):
        super().__init__()
        dilation_1 = 2 ** (n_layers - 1 - layer_idx)
        self.conv_dilated_1 = nn.Conv1d(
            in_channels, in_channels, 3, padding=dilation_1, dilation=dilation_1
        )
        dilation_2 = 2**layer_idx
        self.conv_dilated_2 = nn.Conv1d(
            in_channels, in_channels, 3, padding=dilation_2, dilation=dilation_2
        )
        self.conv_fusion = nn.Conv1d(2 * in_channels, in_channels, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        # x: (N, C_in, L)
        out = torch.cat([self.conv_dilated_1(x), self.conv_dilated_2(x)], dim=1)
        out = self.act(self.conv_fusion(out))
        return self.dropout(out) + x


class DilatedResidualLayers(nn.Module):
    def __init__(self, in_channels: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(in_channels, i) for i in range(n_layers)]
        )

    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out


class DualDilatedResidualLayers(nn.Module):
    def __init__(self, in_channels: int, n_layers: int):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                DualDilatedResidualLayer(in_channels, i, n_layers)
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        return self.layers(x)
