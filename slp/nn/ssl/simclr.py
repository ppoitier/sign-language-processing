from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from slp.core.registry import SSL_METHOD_REGISTRY
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
        self.temperature = temperature
        self.n_views = n_views

    def forward(
        self,
        encoder: SSLEncoder,
        views: Sequence[Tensor],
        masks: Sequence[Optional[Tensor]],
    ) -> SSLOutput:
        embeddings = [encoder.embed(view, mask) for view, mask in zip(views, masks)]
        loss, accuracy = self.nt_xent(embeddings)
        return SSLOutput(
            losses={"total_loss": loss},
            embeddings=embeddings,
            metrics={"contrastive_accuracy": accuracy},
        )

    def nt_xent(self, embeddings: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        batch_size = embeddings[0].size(0)
        n_views = len(embeddings)

        # (n_views * B, E), ordered view-major: [v0 samples..., v1 samples...].
        features = F.normalize(torch.cat(list(embeddings), dim=0), dim=-1)
        similarity = features @ features.t() / self.temperature

        sample_ids = torch.arange(batch_size, device=features.device).repeat(n_views)
        same_sample = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)
        self_mask = torch.eye(
            same_sample.size(0), dtype=torch.bool, device=features.device
        )
        positive_mask = same_sample & ~self_mask

        # Exclude self-similarity from both the numerator and the denominator.
        neg_inf = torch.finfo(similarity.dtype).min
        similarity = similarity.masked_fill(self_mask, neg_inf)

        log_prob = similarity - similarity.logsumexp(dim=1, keepdim=True)
        n_positives = positive_mask.sum(dim=1).clamp(min=1)
        mean_positive_log_prob = (log_prob * positive_mask).sum(dim=1) / n_positives
        loss = -mean_positive_log_prob.mean()

        with torch.no_grad():
            # Fraction of anchors whose most similar neighbour is a positive.
            top1 = similarity.argmax(dim=1)
            accuracy = positive_mask.gather(1, top1.unsqueeze(1)).float().mean()

        return loss, accuracy
