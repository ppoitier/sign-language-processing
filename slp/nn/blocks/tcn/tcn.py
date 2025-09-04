from torch import nn

from slp.nn.blocks.tcn.dilated_layers import DilatedResidualLayer
from slp.nn.blocks.multi_stage import MultiStageModel


class SingleStageTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.dilated_layers = nn.ModuleList([
            DilatedResidualLayer(in_channels=hidden_channels, dilation=2**i)
            for i in range(n_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, mask):
        # x: (N, C_in, L)
        # mask: (N, 1, L)
        out = self.conv_in(x)
        for layer in self.dilated_layers:
            out = layer(out, mask)
        out = self.conv_out(out)
        return out * mask


class MultiStageTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_stages: int, n_layers: int):
        super().__init__()

        stages = [SingleStageTCN(in_channels, hidden_channels, out_channels, n_layers)]
        stages += [
            SingleStageTCN(out_channels, hidden_channels, out_channels, n_layers)
            for _ in range(n_stages-1)
        ]
        self.ms_model = MultiStageModel(stages, activation=nn.Softmax(dim=1))

    def forward(self, x, mask):
        """
        Args:
            x: tensor of shape (N, C_in, T)
            mask: tensor of shape (N, 1, T)

        Returns:
            logits: tensor of shape (N_layers, N, C_in, T)
        """
        return self.ms_model(x, mask)
