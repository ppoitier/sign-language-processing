from typing import Literal

from torch import nn, Tensor

from slp.core.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("projection")
class ProjectionHead(nn.Module):
    """Pooling + MLP projector, as used by SimCLR, SimSiam and BYOL.

    Takes temporal backbone features ``(N, C, T)``, pools them over time and
    maps them to the embedding the self-supervised objective is computed on,
    ``(N, E)``. Inputs that are already pooled, ``(N, C)``, pass straight to
    the MLP, so the same head works with backbones that pool internally.

    Args:
        in_features: channel dimension of the backbone features.
        out_features: embedding dimension the objective sees.
        hidden_features: width of the hidden layers.
        n_layers: total number of linear layers. 2 is SimCLR/BYOL, 3 is SimSiam.
        pooling: how to collapse the temporal axis.
        use_batch_norm: batch-normalise the hidden layers. All three papers do.
        final_batch_norm: append a BatchNorm after the last linear layer.
            SimSiam uses one (without affine parameters); SimCLR and BYOL don't.
        normalize_output: L2-normalise the embedding. Usually left off, since
            the losses normalise internally.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 128,
        hidden_features: int = 512,
        n_layers: int = 2,
        pooling: Literal["mean", "max", "none"] = "mean",
        use_batch_norm: bool = True,
        final_batch_norm: bool = False,
        normalize_output: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"A projector needs at least 1 layer, got {n_layers}.")

        self.pooling = pooling
        self.normalize_output = normalize_output

        layers: list[nn.Module] = []
        current_features = in_features
        for _ in range(n_layers - 1):
            layers.append(
                nn.Linear(current_features, hidden_features, bias=not use_batch_norm)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.ReLU(inplace=True))
            current_features = hidden_features
        layers.append(nn.Linear(current_features, out_features))
        if final_batch_norm:
            layers.append(nn.BatchNorm1d(out_features, affine=False))

        self.projection = nn.Sequential(*layers)

    def pool(self, x: Tensor) -> Tensor:
        if x.dim() == 2 or self.pooling == "none":
            return x
        x = x.flatten(2)  # (N, C, ...) -> (N, C, T)
        if self.pooling == "max":
            return x.max(dim=-1).values
        return x.mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        features = self.projection(self.pool(x))
        if self.normalize_output:
            features = nn.functional.normalize(features, dim=-1)
        return features
