import torch
from torch import nn, Tensor


class GeneralizedIoU(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Reference:
            Hamid Rezatofighi et al.: Generalized Intersection over Union:
            A Metric and A Loss for Bounding Box Regression:
            https://arxiv.org/abs/1902.09630
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_offsets: Tensor, gt_offsets: Tensor) -> Tensor:
        """
        Args:
            pred_offsets: tensor of shape (N, L, 2)
            gt_offsets: tensor of shape (N, L, 2)
        """
        mask = ~(gt_offsets < 0).any(dim=-1)

        # Convert offsets to intervals
        pred_starts, pred_ends = -pred_offsets[:, :, 0], pred_offsets[:, :, 1]
        gt_starts, gt_ends = -gt_offsets[:, :, 0], gt_offsets[:, :, 1]

        # Compute intersection
        inter_starts = torch.maximum(pred_starts, gt_starts)
        inter_ends = torch.minimum(pred_ends, gt_ends)
        intersections = torch.clamp(inter_ends - inter_starts, min=0)

        # Compute union
        pred_lengths = pred_ends - pred_starts
        gt_lengths = gt_ends - gt_starts
        unions = pred_lengths + gt_lengths - intersections

        # Compute gIoU as IoU because C = U in this setting.
        iou = intersections / unions.clamp(min=self.eps)
        loss = (1.0 - iou) * mask

        if self.reduction == 'mean':
            loss = loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
