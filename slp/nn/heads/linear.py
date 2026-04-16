from torch import nn

from slp.core.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("linear")
class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Conv1d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        return self.fc(x)
