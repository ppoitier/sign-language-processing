import torch
from torch import nn, Tensor

from slp.nn.blocks.tcn.dilated_layers import DilatedResidualLayers, DualDilatedResidualLayers


class SingleStageTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.dilated_layers = DilatedResidualLayers(hidden_channels, n_layers)
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, mask):
        # x: (N, C_in, L)
        # mask: (N, 1, L)
        out = self.conv_in(x)
        out = self.dilated_layers(out, mask)
        out = self.conv_out(out)
        return out * mask


class MultiStageTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_stages: int, n_layers: int):
        super().__init__()
        self.first_stage = SingleStageTCN(in_channels, hidden_channels, out_channels, n_layers)
        self.stages = nn.ModuleList([
            SingleStageTCN(out_channels, hidden_channels, out_channels, n_layers)
            for _ in range(n_stages-1)
        ])

    def forward(self, x, mask):
        # x: (N, C_in, L)
        # mask: (N, 1, L)
        out = self.first_stage(x, mask)
        outs = (out, )
        for stage in self.stages:
            out = stage(out.softmax(dim=1) * mask, mask)
            outs += (out, )
        outs = torch.stack(outs)
        return outs


class MultiStageRefinedTCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 n_prediction_layers: int, n_refinement_layers: int, n_refinement_stages: int):
        super().__init__()

        self.prediction_generation_stage = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            DualDilatedResidualLayers(hidden_channels, n_prediction_layers),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )

        self.refinement_stages = nn.Sequential(*[
            SingleStageTCN(out_channels, hidden_channels, out_channels, n_refinement_layers)
            for _ in range(n_refinement_stages)
        ])

    def forward(self, x):
        # x: (N, C_in, L)
        out: Tensor = self.prediction_generation_stage(x)
        outs = (out, )
        for refinement_stage in self.refinement_stages:
            out = refinement_stage(out.softmax(dim=1))
            outs += (out, )
        outs = torch.stack(outs)
        return outs
