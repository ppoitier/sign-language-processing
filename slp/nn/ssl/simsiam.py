from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from slp.core.registry import SSL_METHOD_REGISTRY
from slp.nn.ssl.base import SSLEncoder, SSLMethod, SSLOutput
from slp.nn.ssl.predictor import PredictionHead


@SSL_METHOD_REGISTRY.register("simsiam")
class SimSiam(SSLMethod):
    """SimSiam: negative cosine similarity with a predictor and stop-gradient.

    Both views go through the same encoder; one branch is passed through a
    predictor MLP and regressed onto the *stop-gradiented* embedding of the
    other. No negatives, no momentum encoder — the stop-gradient alone prevents
    collapse (Chen & He, 2021).

    With more than two views the loss averages over every ordered pair, which
    for ``n_views=2`` is the symmetrised loss of the paper.

    Args:
        embedding_dim: output dimension of the model's projection head.
        predictor_hidden_dim: bottleneck width of the predictor. Defaults to
            ``embedding_dim // 4``.
        n_views: number of augmented views per sample.
    """

    def __init__(
        self,
        embedding_dim: int,
        predictor_hidden_dim: int | None = None,
        n_views: int = 2,
    ):
        super().__init__()
        if n_views < 2:
            raise ValueError(f"SimSiam needs at least 2 views, got {n_views}.")
        self.n_views = n_views
        self.predictor = PredictionHead(
            in_channels=embedding_dim,
            hidden_channels=predictor_hidden_dim,
            out_channels=embedding_dim,
        )

    def forward(
        self,
        encoder: SSLEncoder,
        views: Sequence[Tensor],
        masks: Sequence[Optional[Tensor]],
    ) -> SSLOutput:
        embeddings = [encoder.embed(view, mask) for view, mask in zip(views, masks)]
        predictions = [self.predictor(z) for z in embeddings]

        loss = symmetric_negative_cosine(predictions, embeddings)
        return SSLOutput(
            losses={"total_loss": loss},
            embeddings=embeddings,
            metrics={"output_std": normalized_output_std(embeddings)},
        )


def symmetric_negative_cosine(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
) -> Tensor:
    """Mean negative cosine similarity over every ordered pair of views.

    ``targets`` are detached, which is the stop-gradient that keeps the
    objective from collapsing to a constant.
    """
    losses = []
    for i, prediction in enumerate(predictions):
        for j, target in enumerate(targets):
            if i == j:
                continue
            losses.append(-F.cosine_similarity(prediction, target.detach(), dim=-1).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def normalized_output_std(embeddings: Sequence[Tensor]) -> Tensor:
    """SimSiam's collapse indicator, scaled so that healthy training gives ~1.

    The paper tracks the per-dimension standard deviation of the L2-normalised
    embeddings, which sits around ``1/sqrt(d)`` when the representation is
    spread out and drops to 0 when it collapses. Multiplying by ``sqrt(d)``
    makes the healthy value 1 regardless of the embedding size.
    """
    stacked = torch.cat([F.normalize(z, dim=-1) for z in embeddings], dim=0)
    return stacked.std(dim=0).mean() * (stacked.size(-1) ** 0.5)
