from torch import nn, Tensor


class ProjectionHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 128,
            hidden_channels: int = 512,
            normalize_output: bool = True,
    ):
        super(ProjectionHead, self).__init__()

        self.normalize_output = normalize_output
        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x of shape (B, E)
        feat = self.projection(x)
        if self.normalize_output:
            feat = nn.functional.normalize(feat, dim=-1)
        return feat