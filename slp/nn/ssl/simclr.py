from typing import Optional, Sequence

from torch import Tensor

from slp.core.registry import SSL_METHOD_REGISTRY
from slp.nn.losses.contrastive.nt_xent import NTXentLoss
from slp.nn.ssl.base import SSLEncoder, SSLMethod, SSLOutput


@SSL_METHOD_REGISTRY.register("simclr")
class SimCLR(SSLMethod):
    """SimCLR / NT-Xent contrastive objective.

    Every view of a sample is an anchor whose positives are the other views of
    that same sample; every view of every other sample in the batch is a
    negative. With ``n_views=2`` this is exactly NT-Xent (Chen et al., 2020);
    with more views it is the multi-positive form also used by SupCon, which
    reduces to NT-Xent when there is a single positive.

    This method holds no parameters of its own: the projector is part of the
    model, declared as the ``projection`` head in the model config.

    Args:
        temperature: softmax temperature. Lower values weight hard negatives
            more heavily. 0.1-0.2 is the usual range.
        n_views: number of augmented views per sample.
    """

    def __init__(self, temperature: float = 0.1, n_views: int = 2):
        super().__init__()
        if n_views < 2:
            raise ValueError(f"SimCLR needs at least 2 views, got {n_views}.")
        self.n_views = n_views
        self.nt_xent = NTXentLoss(temperature=temperature)

    def forward(
        self,
        encoder: SSLEncoder,
        views: Sequence[Tensor],
        masks: Sequence[Optional[Tensor]],
    ) -> SSLOutput:
        embeddings = [encoder.embed(view, mask) for view, mask in zip(views, masks)]
        loss, accuracy = self.nt_xent(embeddings, return_accuracy=True)
        return SSLOutput(
            losses={"total_loss": loss},
            embeddings=embeddings,
            metrics={"contrastive_accuracy": accuracy},
        )
