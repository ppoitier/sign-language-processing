"""
NT-Xent (SimCLR) loss for V >= 2 aligned views, without labels.

With two views each anchor has exactly one positive (its counterpart in the
other view), so the NT-Xent denominator (that positive plus all negatives) is
simply every off-diagonal entry of the row, making each row an ordinary
softmax classification, i.e. cross entropy. That shortcut is specific to
V=2. With V>2 views, each anchor has V-1 positives (all other views of the
same sample), so instead of a single-label cross entropy we use the general
multi-positive NT-Xent form: for each anchor, average the log-softmax over
its positive set (this is the "mean of log" variant used e.g. in SupCon).
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def nt_xent_loss(views: list[Tensor], temperature=0.07, return_accuracy=False):
    """NT-Xent loss for V aligned views.

    Args:
        views: V tensors of shape (N, D), aligned so that views[k][i] all
                come from the same source sample. Do not need to be
                pre-normalized.
        temperature: scalar T.
        return_accuracy: also return the contrastive accuracy, i.e. the
            fraction of anchors whose nearest neighbour (by cosine
            similarity, self excluded) is a true positive. A cheap,
            interpretable read on the same similarity matrix, useful for
            spotting collapse without an extra forward pass.

    Returns:
        Scalar loss, the mean over all V*N anchors. If ``return_accuracy``,
        a ``(loss, accuracy)`` tuple instead.
    """
    num_views = len(views)
    n = views[0].shape[0]
    z = F.normalize(torch.cat(views, dim=0), dim=1)
    nv = z.shape[0]

    logits = (z @ z.t()) / temperature
    logits.fill_diagonal_(float("-inf"))  # an anchor is not its own candidate

    # Row i's positives are every row j != i with j % n == i % n (same
    # sample, other view).
    sample_id = torch.arange(nv, device=z.device) % n
    positive_mask = sample_id.unsqueeze(0) == sample_id.unsqueeze(1)
    positive_mask.fill_diagonal_(False)

    log_prob = F.log_softmax(logits, dim=1)
    log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_anchor = -log_prob.sum(dim=1) / (num_views - 1)
    loss = loss_per_anchor.mean()

    if not return_accuracy:
        return loss

    with torch.no_grad():
        top1 = logits.argmax(dim=1)
        accuracy = positive_mask.gather(1, top1.unsqueeze(1)).float().mean()
    return loss, accuracy


class NTXentLoss(torch.nn.Module):
    """Module wrapper, so it can live in your model / be moved with .to()."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, views: list[Tensor], return_accuracy=False):
        return nt_xent_loss(
            views, temperature=self.temperature, return_accuracy=return_accuracy
        )

    def extra_repr(self):
        return f"temperature={self.temperature}"