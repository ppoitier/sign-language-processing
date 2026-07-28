from torch import nn

from slp.core.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("linear")
class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Conv1d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(-1)  # (N, C) -> (N, C, 1)

        out = self.fc(x)

        if squeeze_output:
            out = out.squeeze(-1)  # (N, C_out, 1) -> (N, C_out)
        return out
