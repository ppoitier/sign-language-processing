from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.no_grad()
def representation_statistics(
    embeddings: Sequence[Tensor],
    prefix: str = "",
) -> dict[str, Tensor]:
    """Label-free diagnostics for a batch of self-supervised embeddings.

    Self-supervised losses are poor progress indicators: a collapsed encoder
    that maps everything to one point reaches a low SimSiam/BYOL loss. These
    statistics tell collapse apart from genuine learning without needing any
    labels.

    Metrics, all computed on L2-normalised embeddings:
        ``repr_std``: mean per-dimension standard deviation, rescaled by
            ``sqrt(d)`` so that ~1 is healthy and ~0 means collapse.
        ``pos_cosine_sim``: mean cosine similarity between views of the same
            sample. Should rise during training, but reaching 1 while
            ``repr_std`` falls is collapse.
        ``neg_cosine_sim``: mean cosine similarity between different samples.
            Should stay near 0.
        ``alignment``: mean squared distance between positive pairs
            (lower is better; Wang & Isola, 2020).
        ``uniformity``: log of the mean Gaussian potential between different
            samples (lower means the embeddings spread more evenly on the
            hypersphere).

    Args:
        embeddings: one ``(B, E)`` tensor per view.
        prefix: prepended to every metric name, e.g. ``"validation/"``.

    Returns:
        A dict of scalar tensors, ready for ``TrainerBase.log_metrics``.
    """
    if not embeddings:
        return {}

    normalized = [F.normalize(z.detach().float(), dim=-1) for z in embeddings]
    stacked = torch.cat(normalized, dim=0)
    embedding_dim = stacked.size(-1)

    metrics: dict[str, Tensor] = {
        f"{prefix}repr_std": stacked.std(dim=0).mean() * (embedding_dim**0.5),
    }

    # Positive pairs: same sample, different view.
    if len(normalized) > 1:
        positive_similarities = [
            (normalized[i] * normalized[j]).sum(-1).mean()
            for i in range(len(normalized))
            for j in range(i + 1, len(normalized))
        ]
        positive_similarity = torch.stack(positive_similarities).mean()
        metrics[f"{prefix}pos_cosine_sim"] = positive_similarity
        metrics[f"{prefix}alignment"] = 2 - 2 * positive_similarity

    # Negative pairs: distinct samples within the first view.
    view = normalized[0]
    batch_size = view.size(0)
    if batch_size > 1:
        similarity = view @ view.t()
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=view.device)
        negative_similarity = similarity[off_diagonal]
        metrics[f"{prefix}neg_cosine_sim"] = negative_similarity.mean()
        # ||a - b||^2 = 2 - 2 * cos for unit vectors.
        squared_distances = (2 - 2 * negative_similarity).clamp(min=0)
        metrics[f"{prefix}uniformity"] = torch.log(
            torch.exp(-2 * squared_distances).mean()
        )

    return metrics
