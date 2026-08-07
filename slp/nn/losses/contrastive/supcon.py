"""
Supervised Contrastive loss (Khosla et al. 2020), the L_out variant.

Numerically equivalent to pytorch_metric_learning.losses.SupConLoss with its
default settings (CosineSimilarity distance, AvgNonZeroReducer), but
self-contained.

For each anchor i with positive set P(i) (same label, excluding itself):

                 1            exp(sim(i, p) / T)
    L_i = - ---------- SUM log ------------------------
              |P(i)|     p     SUM  exp(sim(i, k) / T)
                                k != i

Two differences from NT-Xent: the denominator runs over *every* other sample
(other positives included, not just the negatives), and the average is taken
per anchor first and then over anchors, rather than over positive pairs
directly. When every anchor has exactly one positive -- the SimCLR case -- both
differences vanish and this reduces to NT-Xent exactly.
"""

import torch
import torch.nn.functional as F


def sup_con_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss over a batch of embeddings.

    Args:
        embeddings: (B, D) float tensor. Does not need to be pre-normalized.
        labels:     (B,) integer tensor.
        temperature: scalar T.

    Returns:
        Scalar loss, averaged over the anchors that actually have a positive.
        Anchors whose class is a singleton in the batch contribute nothing.
        Returns 0 if the batch has no positive pairs or no negative pairs.
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim = (embeddings @ embeddings.t()) / temperature

    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=sim.device)
    pos_mask = same & ~eye

    if not (pos_mask.any() and (~same).any()):
        return sim.sum() * 0  # keeps the graph / dtype / device intact

    # Denominator: everything except the anchor itself, i.e. positives U negatives.
    logits = sim.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    n_pos = pos_mask.sum(dim=1)
    log_prob_pos = log_prob.masked_fill(~pos_mask, 0).sum(dim=1)
    per_anchor = -log_prob_pos / n_pos.clamp(min=1)
    return per_anchor[n_pos > 0].mean()


class SupConLoss(torch.nn.Module):
    """Module wrapper, so it can live in your model / be moved with .to()."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        return sup_con_loss(embeddings, labels, self.temperature)

    def extra_repr(self):
        return f"temperature={self.temperature}"