from torch import nn, Tensor


class PredictionHead(nn.Module):
    """Bottleneck MLP mapping an embedding onto another embedding.

    This is the asymmetry that keeps SimSiam and BYOL from collapsing: one
    branch predicts the other, which is stop-gradiented. Both papers use a
    hidden layer much narrower than the embedding (256 for a 2048-d embedding).

    Args:
        in_channels: embedding dimension produced by the projector.
        hidden_channels: bottleneck width. Defaults to ``in_channels // 4``.
        out_channels: output dimension. Defaults to ``in_channels``, which is
            required since the prediction is compared to an embedding.
        use_batch_norm: whether to normalise the bottleneck activations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        hidden_channels = hidden_channels or max(in_channels // 4, 1)
        out_channels = out_channels or in_channels

        layers: list[nn.Module] = [nn.Linear(in_channels, hidden_channels, bias=not use_batch_norm)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_channels, out_channels))

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)
